using Azure.Identity;
using Azure.AI.Projects;
using System.Text.Json;

using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

using Azure.Monitor.OpenTelemetry.Exporter;

using OpenTelemetry;
using OpenTelemetry.Resources;
using OpenTelemetry.Trace;

using FactoryWorkflow;

// ============================================================================
// Application Startup
// ============================================================================

DotNetEnv.Env.TraversePath().Load();

var builder = WebApplication.CreateBuilder(args);

// Configure services
builder.Services.AddHttpClient();
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
        policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod());
});
builder.Configuration.AddEnvironmentVariables();

// Register Azure AI Project client for Agent Service access
builder.Services.AddSingleton(sp =>
{
    var endpoint = sp.GetRequiredService<IConfiguration>()["AZURE_AI_PROJECT_ENDPOINT"]
        ?? throw new InvalidOperationException("AZURE_AI_PROJECT_ENDPOINT is required");
    return new AIProjectClient(new Uri(endpoint), new DefaultAzureCredential());
});

builder.Services.AddSingleton<ILoggerFactory>(sp => LoggerFactory.Create(b => b.AddConsole()));

// Configure OpenTelemetry tracing
ConfigureTracing(builder);

var app = builder.Build();
app.UseCors();

// ============================================================================
// API Endpoints
// ============================================================================

app.MapGet("/health", () => Results.Ok(new { Status = "Healthy", Timestamp = DateTimeOffset.UtcNow }));

app.MapPost("/api/analyze_machine", async (
    AnalyzeRequest request,
    AIProjectClient projectClient,
    IConfiguration config,
    ILoggerFactory loggerFactory,
    ILogger<Program> logger) =>
{
    logger.LogInformation("Starting analysis for machine {MachineId}", request.machine_id);

    try
    {
        // ================================================================
        // Step 1: Collect all agents for the workflow pipeline
        // ================================================================
        // The workflow executes agents sequentially, passing text output
        // from each agent to the next. Agent order matters!
        //
        // Pipeline: AnomalyClassification → FaultDiagnosis → RepairPlanner
        //           → MaintenanceScheduler → PartsOrdering
        // ================================================================

        var agents = new List<AIAgent>();

        // Agent Service agents (Azure AI Foundry hosted)
        agents.AddRange(await AgentServiceProvider.GetAgentsAsync(projectClient, logger));

        // Local agent with Cosmos DB tools
        var repairPlanner = LocalAgentProvider.GetRepairPlannerAgent(config, loggerFactory, logger);
        if (repairPlanner != null) agents.Add(repairPlanner);

        // A2A agents (Python services from Challenge 3)
        agents.AddRange(await A2AAgentProvider.GetAgentsAsync(config, logger));

        logger.LogInformation("Workflow pipeline: [{Agents}]", string.Join(" → ", agents.Select(a => a.Name)));

        // ================================================================
        // Step 2: Build and execute the workflow
        // ================================================================
        // Uses AgentWorkflowBuilder.BuildSequential for native SDK support.
        // The latest SDK version handles conversation history properly.
        // ================================================================

        var telemetryJson = JsonSerializer.Serialize(request);
        var workflowResult = await ExecuteWorkflowAsync(agents, telemetryJson, logger);

        return Results.Ok(workflowResult);
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Workflow failed for machine {MachineId}", request.machine_id);
        return Results.Problem(ex.Message);
    }
});

// ============================================================================
// Streaming Endpoint (Server-Sent Events)
// ============================================================================

app.MapPost("/api/analyze_machine/stream", async (
    HttpContext httpContext,
    AIProjectClient projectClient,
    IConfiguration config,
    ILoggerFactory loggerFactory,
    ILogger<Program> logger) =>
{
    // Read request body manually since we're using HttpContext directly
    var request = await httpContext.Request.ReadFromJsonAsync<AnalyzeRequest>();
    if (request == null)
    {
        httpContext.Response.StatusCode = 400;
        await httpContext.Response.WriteAsync("Invalid request body");
        return;
    }

    logger.LogInformation("Starting SSE stream for machine {MachineId}", request.machine_id);

    // Set up SSE response headers
    httpContext.Response.ContentType = "text/event-stream";
    httpContext.Response.Headers.CacheControl = "no-cache";
    httpContext.Response.Headers.Connection = "keep-alive";

    var jsonOptions = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    // Helper to send SSE events
    async Task SendEventAsync<T>(T evt) where T : SseEvent
    {
        var json = JsonSerializer.Serialize(evt, jsonOptions);
        await httpContext.Response.WriteAsync($"data: {json}\n\n");
        await httpContext.Response.Body.FlushAsync();
    }

    try
    {
        // Collect agents (same as non-streaming endpoint)
        var agents = new List<AIAgent>();
        agents.AddRange(await AgentServiceProvider.GetAgentsAsync(projectClient, logger));
        
        var repairPlanner = LocalAgentProvider.GetRepairPlannerAgent(config, loggerFactory, logger);
        if (repairPlanner != null) agents.Add(repairPlanner);
        
        agents.AddRange(await A2AAgentProvider.GetAgentsAsync(config, logger));

        // Send workflow_started event
        await SendEventAsync(new SseWorkflowStarted
        {
            AgentPipeline = agents.Select(a => a.Name).ToList()
        });

        // Track agent index for progress updates
        int agentIndex = 0;

        // Send first agent_started event
        if (agents.Count > 0)
        {
            await SendEventAsync(new SseAgentStarted
            {
                AgentName = agents[0].Name,
                AgentIndex = 0
            });
        }

        // Execute workflow with streaming events
        var telemetryJson = JsonSerializer.Serialize(request);
        var workflowResult = await ExecuteWorkflowStreamingAsync(
            agents, 
            telemetryJson, 
            logger,
            async (step) =>
            {
                await SendEventAsync(new SseAgentCompleted { Step = step });
                agentIndex++;
                
                // Send agent_started for next agent if there is one
                if (agentIndex < agents.Count)
                {
                    await SendEventAsync(new SseAgentStarted
                    {
                        AgentName = agents[agentIndex].Name,
                        AgentIndex = agentIndex
                    });
                }
            });

        // Send final workflow_completed event
        await SendEventAsync(new SseWorkflowCompleted { Result = workflowResult });
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "SSE workflow failed for machine {MachineId}", request.machine_id);
        await SendEventAsync(new SseWorkflowError { Error = ex.Message });
    }
});

app.Run();

// ============================================================================
// Workflow Execution
// ============================================================================

static async Task<WorkflowResponse> ExecuteWorkflowAsync(
    List<AIAgent> agents,
    string input,
    ILogger logger)
{
    // Build sequential workflow using the built-in AgentWorkflowBuilder.
    // The SDK now properly handles conversation history between agents,
    // eliminating the need for the TextOnlyAgentExecutor workaround.
    var workflow = AgentWorkflowBuilder.BuildSequential(agents);
    logger.LogInformation("Built sequential workflow with {Count} agents", agents.Count);

    // Prepare input as a ChatMessage (required by agent workflows)
    var messages = new List<ChatMessage> { new(ChatRole.User, input) };

    // Execute the workflow and collect events
    var run = await InProcessExecution.Default.StreamAsync(workflow, messages);

    // Collect step results and final output
    var agentSteps = new List<AgentStepResult>();
    string? finalOutput = null;
    string? currentAgentName = null;
    AgentStepResult? currentStep = null;

    await foreach (var evt in run.WatchStreamAsync())
    {
        switch (evt)
        {
            case ExecutorInvokedEvent invoked:
                // New agent starting - create a step result
                currentAgentName = invoked.ExecutorId;
                currentStep = new AgentStepResult { AgentName = currentAgentName ?? "Unknown" };
                logger.LogDebug("Agent {AgentName} invoked", currentAgentName);
                break;

            case AgentResponseEvent responseEvent:
                // Collect agent response data using the typed Response property
                if (currentStep != null && responseEvent.Response.Messages != null)
                {
                    foreach (var msg in responseEvent.Response.Messages)
                    {
                        if (msg.Role == ChatRole.Assistant)
                        {
                            foreach (var content in msg.Contents)
                            {
                                if (content is TextContent tc && !string.IsNullOrWhiteSpace(tc.Text))
                                {
                                    currentStep.TextOutput += tc.Text;
                                    currentStep.FinalMessage = tc.Text;
                                    finalOutput = tc.Text; // Keep updating with latest
                                }
                                else if (content is FunctionCallContent fcc)
                                {
                                    currentStep.ToolCalls.Add(new ToolCallInfo
                                    {
                                        ToolName = fcc.Name,
                                        CallId = fcc.CallId,
                                        Arguments = JsonSerializer.Serialize(fcc.Arguments)
                                    });
                                }
                            }
                        }
                        else if (msg.Role == ChatRole.Tool)
                        {
                            foreach (var content in msg.Contents)
                            {
                                if (content is FunctionResultContent frc)
                                {
                                    var matchingCall = currentStep.ToolCalls.LastOrDefault(t => t.CallId == frc.CallId);
                                    if (matchingCall != null)
                                    {
                                        var resultStr = frc.Result?.ToString();
                                        matchingCall.Result = resultStr?.Substring(0, Math.Min(500, resultStr.Length));
                                    }
                                }
                            }
                        }
                    }
                }
                break;

            case ExecutorCompletedEvent completed:
                // Agent completed - save the step
                if (currentStep != null)
                {
                    logger.LogDebug("Agent {AgentName} completed with {ToolCallCount} tool calls", 
                        currentStep.AgentName, currentStep.ToolCalls.Count);
                    agentSteps.Add(currentStep);
                    currentStep = null;
                }
                break;

            case WorkflowOutputEvent outputEvent:
                // Capture final workflow output
                if (outputEvent.Data is AgentResponse agentResponse)
                {
                    foreach (var msg in agentResponse.Messages ?? [])
                    {
                        if (msg.Role == ChatRole.Assistant)
                        {
                            foreach (var content in msg.Contents)
                            {
                                if (content is TextContent tc && !string.IsNullOrWhiteSpace(tc.Text))
                                {
                                    finalOutput = tc.Text;
                                }
                            }
                        }
                    }
                }
                break;
        }
    }

    logger.LogInformation("Workflow completed with {StepCount} agent steps", agentSteps.Count);

    return new WorkflowResponse
    {
        AgentSteps = agentSteps,
        FinalMessage = finalOutput ?? agentSteps.LastOrDefault()?.FinalMessage
    };
}

/// <summary>
/// Executes the workflow with streaming support, calling the onStepCompleted callback after each agent finishes.
/// </summary>
static async Task<WorkflowResponse> ExecuteWorkflowStreamingAsync(
    List<AIAgent> agents,
    string input,
    ILogger logger,
    Func<AgentStepResult, Task> onStepCompleted)
{
    // Build sequential workflow using the built-in AgentWorkflowBuilder
    var workflow = AgentWorkflowBuilder.BuildSequential(agents);
    logger.LogInformation("Built sequential workflow with {Count} agents for streaming", agents.Count);

    // Prepare input as a ChatMessage
    var messages = new List<ChatMessage> { new(ChatRole.User, input) };

    // Execute the workflow and collect events with streaming callbacks
    var run = await InProcessExecution.Default.StreamAsync(workflow, messages);

    // Collect step results and final output
    var agentSteps = new List<AgentStepResult>();
    string? finalOutput = null;
    string? currentAgentName = null;
    AgentStepResult? currentStep = null;

    await foreach (var evt in run.WatchStreamAsync())
    {
        switch (evt)
        {
            case ExecutorInvokedEvent invoked:
                currentAgentName = invoked.ExecutorId;
                currentStep = new AgentStepResult { AgentName = currentAgentName ?? "Unknown" };
                logger.LogDebug("Agent {AgentName} invoked (streaming)", currentAgentName);
                break;

            case AgentResponseEvent responseEvent:
                if (currentStep != null && responseEvent.Response.Messages != null)
                {
                    foreach (var msg in responseEvent.Response.Messages)
                    {
                        if (msg.Role == ChatRole.Assistant)
                        {
                            foreach (var content in msg.Contents)
                            {
                                if (content is TextContent tc && !string.IsNullOrWhiteSpace(tc.Text))
                                {
                                    currentStep.TextOutput += tc.Text;
                                    currentStep.FinalMessage = tc.Text;
                                    finalOutput = tc.Text;
                                }
                                else if (content is FunctionCallContent fcc)
                                {
                                    currentStep.ToolCalls.Add(new ToolCallInfo
                                    {
                                        ToolName = fcc.Name,
                                        CallId = fcc.CallId,
                                        Arguments = JsonSerializer.Serialize(fcc.Arguments)
                                    });
                                }
                            }
                        }
                        else if (msg.Role == ChatRole.Tool)
                        {
                            foreach (var content in msg.Contents)
                            {
                                if (content is FunctionResultContent frc)
                                {
                                    var matchingCall = currentStep.ToolCalls.LastOrDefault(t => t.CallId == frc.CallId);
                                    if (matchingCall != null)
                                    {
                                        var resultStr = frc.Result?.ToString();
                                        matchingCall.Result = resultStr?.Substring(0, Math.Min(500, resultStr.Length));
                                    }
                                }
                            }
                        }
                    }
                }
                break;

            case ExecutorCompletedEvent completed:
                if (currentStep != null)
                {
                    logger.LogDebug("Agent {AgentName} completed (streaming) with {ToolCallCount} tool calls", 
                        currentStep.AgentName, currentStep.ToolCalls.Count);
                    agentSteps.Add(currentStep);
                    
                    // Notify the callback
                    await onStepCompleted(currentStep);
                    currentStep = null;
                }
                break;

            case WorkflowOutputEvent outputEvent:
                if (outputEvent.Data is AgentResponse agentResponse)
                {
                    foreach (var msg in agentResponse.Messages ?? [])
                    {
                        if (msg.Role == ChatRole.Assistant)
                        {
                            foreach (var content in msg.Contents)
                            {
                                if (content is TextContent tc && !string.IsNullOrWhiteSpace(tc.Text))
                                {
                                    finalOutput = tc.Text;
                                }
                            }
                        }
                    }
                }
                break;
        }
    }

    logger.LogInformation("Streaming workflow completed with {StepCount} agent steps", agentSteps.Count);

    return new WorkflowResponse
    {
        AgentSteps = agentSteps,
        FinalMessage = finalOutput ?? agentSteps.LastOrDefault()?.FinalMessage
    };
}

// ============================================================================
// Telemetry Configuration
// ============================================================================

static void ConfigureTracing(WebApplicationBuilder builder)
{
    const string SourceName = "FactoryWorkflow";

    var resourceBuilder = ResourceBuilder.CreateDefault()
        .AddService(
            serviceName: builder.Environment.ApplicationName,
            serviceVersion: typeof(Program).Assembly.GetName().Version?.ToString())
        .AddAttributes([
            new KeyValuePair<string, object>("deployment.environment", builder.Environment.EnvironmentName),
        ]);

    var tracerBuilder = Sdk.CreateTracerProviderBuilder()
        .SetResourceBuilder(resourceBuilder)
        .AddSource(SourceName, "ChatClient")
        .AddSource("Microsoft.Agents.AI.*") // Agent Framework telemetry
        .AddSource("Microsoft.Extensions.AI.*") // Extensions AI telemetry
        .AddSource("AnomalyClassificationAgent", "FaultDiagnosisAgent", "RepairPlannerAgent")
        .AddSource("MaintenanceSchedulerAgent", "PartsOrderingAgent")
        .AddAspNetCoreInstrumentation()
        .AddHttpClientInstrumentation()
        .AddOtlpExporter();

    var appInsightsConnectionString = builder.Configuration["ApplicationInsights:ConnectionString"];
    if (!string.IsNullOrWhiteSpace(appInsightsConnectionString))
    {
        tracerBuilder.AddAzureMonitorTraceExporter(options =>
            options.ConnectionString = appInsightsConnectionString);
    }

    tracerBuilder.Build();
}
