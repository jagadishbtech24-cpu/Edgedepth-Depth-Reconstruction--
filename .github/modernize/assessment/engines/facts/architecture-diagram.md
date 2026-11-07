# Architecture Diagram

EdgeDepth accepts registered RGB and low-resolution or sparse depth, then compares
classical methods, EdgeGuideNet, a separate DepthUNet branch, and EdgeGuideNet hybrid
refinement. The editable flowchart and high-resolution export are maintained at the
repository root as `architecture_diagram.drawio` and `architecture_diagram.png`.

## Application Architecture

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart TD
    subgraph ClientLayer["Client Layer"]
        Browser["React dashboard"]
    end
    subgraph ApiLayer["API Layer"]
        FastAPI["FastAPI and Uvicorn"]
        Reconstruction["Reconstruction service"]
        ResultCache[("Bounded in-memory result cache")]
    end
    subgraph ProcessingLayer["Depth Processing"]
        Preparation["RGB-D validation and depth preparation"]
        Classical["Nearest, Bilinear, Bicubic, Guided, Joint bilateral"]
        EdgeModel["EdgeGuideNet"]
        Hybrid["EdgeGuideNet hybrid refinement"]
        DepthModel["DepthUNet separate branch"]
        Evaluation["Reference metrics and boundary analysis"]
    end
    subgraph LocalAssets["Local Project Assets"]
        SceneFiles[("NYU and curated RGB-D files")]
        ModelFiles[("Separate model checkpoints")]
    end
    subgraph ResultLayer["Result Views"]
        Views["Depth, boundary, error, and 3D views"]
    end

    Browser -->|"scene selection, settings, or RGB-D upload"| FastAPI
    SceneFiles -->|"registered scene pairs"| FastAPI
    FastAPI -->|"validated arrays and settings"| Reconstruction
    Reconstruction -->|"aligned RGB and prepared depth"| Preparation
    Preparation -->|"RGB and depth samples"| Classical
    Preparation -->|"RGB, depth, and mask"| EdgeModel
    Preparation -->|"RGB, sparse depth, and mask"| DepthModel
    ModelFiles -->|"EdgeGuideNet checkpoint"| EdgeModel
    ModelFiles -->|"DepthUNet checkpoint"| DepthModel
    EdgeModel -->|"depth prediction and RGB"| Hybrid
    Preparation -->|"registered RGB guidance"| Hybrid
    Classical -->|"per-method predictions"| Evaluation
    EdgeModel -->|"per-method prediction"| Evaluation
    Hybrid -->|"refined prediction"| Evaluation
    DepthModel -->|"independent prediction"| Evaluation
    SceneFiles -->|"optional reference depth"| Evaluation
    Evaluation -->|"metrics and prediction views"| ResultCache
    ResultCache -->|"cached result"| Views
    Views -->|"dashboard response and exports"| Browser
```

### Technology Stack Summary

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| Presentation | React | 18.3.1 | Interactive dashboard and result inspection |
| Frontend tooling | Vite | 6.x | Development server and production bundle |
| API | FastAPI and Uvicorn | FastAPI 0.110+; Uvicorn 0.29+ | Upload, scene, reconstruction, and export endpoints |
| Model runtime | PyTorch | 2.6+ | EdgeGuideNet and strictly loaded DepthUNet inference |
| Image and array processing | NumPy, OpenCV, Pillow | See `requirements.txt` | Depth processing, image decoding, and metrics |
| Visualization | Plotly | See `requirements.txt` | Metric charts and interactive 3D depth surfaces |

### Data Storage & External Services

There is no database, message broker, or required external service. Scenes and model
checkpoints are read from local project files. Reconstruction results are held in a
bounded in-memory cache for API downloads and visualization; they expire or are evicted
when capacity is reached.

### Key Architectural Decisions

- Classical baselines, EdgeGuideNet, and DepthUNet produce separate method predictions;
  DepthUNet does not replace EdgeGuideNet.
- Hybrid refinement consumes the EdgeGuideNet prediction and RGB guidance, then applies
  the configured Guided or Joint Bilateral filter and optional boundary sharpening.
- Quantitative metrics and absolute-error maps are only produced when reference depth is
  available; per-method depth, boundary, and 3D outputs do not require a reference.

## Component Relationships

<!-- mermaid-checked: no \n, no em-dash/en-dash, no {} in labels, subgraphs are id["label"], arrows are -->|"label"|, all subgraphs closed by end, ids unique -->
```mermaid
flowchart LR
    subgraph PresentationLayer["Presentation"]
        cReact["React dashboard"]
    end
    subgraph HttpLayer["HTTP API"]
        cRoutes["FastAPI routes"]
        cResponse["Preview and export assembly"]
        cCache[("Result cache")]
    end
    subgraph ServiceLayer["Reconstruction Components"]
        cService["Reconstruction service"]
        cClassical["Classical algorithms"]
        cHybrid["Hybrid pipeline"]
        cEdgeLoader["EdgeGuideNet loader and inference"]
        cDepthLoader["DepthUNet strict loader and inference"]
        cMetrics["Metrics and boundary analysis"]
    end
    subgraph DataLayer["Local Data"]
        cScenes[("NYU and curated scene files")]
        cWeights[("Model checkpoint files")]
    end

    cReact -->|"HTTP and multipart requests"| cRoutes
    cScenes -->|"scene catalog and inputs"| cRoutes
    cRoutes -->|"prepared arrays and settings"| cService
    cService -->|"interpolation and RGB filters"| cClassical
    cService -->|"EdgeGuideNet reconstruction"| cHybrid
    cHybrid -->|"loads EdgeGuideNet"| cEdgeLoader
    cService -->|"separate model inference"| cDepthLoader
    cWeights -->|"EdgeGuideNet weights"| cEdgeLoader
    cWeights -->|"DepthUNet weights"| cDepthLoader
    cService -->|"per-method arrays and reference"| cMetrics
    cMetrics -->|"metrics and boundary scores"| cResponse
    cService -->|"predictions and visual features"| cResponse
    cResponse -->|"result payload"| cCache
    cCache -->|"depth and surface downloads"| cRoutes
    cRoutes -->|"JSON, images, and exports"| cReact
```

### Component Inventory

| Component | Layer | Type | Responsibility |
|---|---|---|---|
| React dashboard | Presentation | Frontend application | Scene selection, upload, experiment controls, comparisons, charts, and 3D view |
| FastAPI routes | HTTP API | REST and multipart endpoints | Health, scene catalog, reconstruction, cached depth export, and surface data |
| Preview and export assembly | HTTP API | Response transformation | Builds depth, boundary, error previews and JSON-safe metric payloads |
| Result cache | HTTP API | Bounded in-memory storage | Retains recent results for depth and surface requests |
| Reconstruction service | Processing | Orchestration service | Prepares data and runs selected reconstruction and evaluation paths |
| Classical algorithms | Processing | Baselines and filters | Nearest, Bilinear, Bicubic, Guided, and Joint Bilateral predictions |
| Hybrid pipeline | Processing | Refinement pipeline | Runs EdgeGuideNet, RGB-depth affinity, filtering, and optional sharpening |
| EdgeGuideNet loader and inference | Processing | Model runtime | Loads the EdgeGuideNet checkpoint and produces depth and boundary confidence |
| DepthUNet strict loader and inference | Processing | Separate model runtime | Loads the supplied five-channel checkpoint with strict state matching |
| Metrics and boundary analysis | Processing | Evaluation utilities | Computes valid-reference metrics and boundary scores |
| NYU and curated scene files | Data | Local image and depth files | Provides paired registered examples and optional references |
| Model checkpoint files | Data | Local model weights | Stores distinct EdgeGuideNet and DepthUNet checkpoints |
