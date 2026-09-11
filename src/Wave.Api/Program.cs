using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddOpenApi();

// CORS for Angular dev server
builder.Services.AddCors(options =>
{
    options.AddPolicy("AngularDev", policy =>
    {
        policy.WithOrigins("http://localhost:4200")
              .AllowAnyHeader()
              .AllowAnyMethod()
              .AllowCredentials();
    });
});

// WaveCore HTTP client with resilience
builder.Services.AddHttpClient("WaveCore", (sp, client) =>
{
    var config = sp.GetRequiredService<IConfiguration>();
    var baseUrl = config["WaveCore:BaseUrl"] ?? "http://localhost:8100";
    client.BaseAddress = new Uri(baseUrl);
    client.Timeout = TimeSpan.FromSeconds(30);
})
.AddStandardResilienceHandler();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseCors("AngularDev");

app.MapGet("/health", async (IHttpClientFactory httpClientFactory, ILogger<Program> logger) =>
{
    var client = httpClientFactory.CreateClient("WaveCore");
    string coreStatus = "unknown";
    string? coreDbStatus = null;

    try
    {
        var response = await client.GetFromJsonAsync<CoreHealthResponse>("/health");
        if (response != null)
        {
            coreStatus = response.Status;
            coreDbStatus = response.Db;
        }
    }
    catch (Exception ex)
    {
        logger.LogWarning(ex, "WaveCore health check failed");
        coreStatus = $"unreachable ({ex.Message})";
    }

    var result = new
    {
        api = "ok",
        core = coreStatus,
        core_db = coreDbStatus,
        timestamp = DateTimeOffset.UtcNow
    };

    return Results.Ok(result);
})
.WithName("HealthCheck");

app.Run();

record CoreHealthResponse(
    [property: JsonPropertyName("status")] string Status,
    [property: JsonPropertyName("version")] string Version,
    [property: JsonPropertyName("db")] string Db
);
