# Sequence Diagram - Video Processing Pipeline

```mermaid
sequenceDiagram
    autonumber
    actor Operator as Operator
    participant Policy as Source Policy
    participant Queue as Approved Queue
    participant Downloader as Safe Downloader
    participant Inspector as Inspector
    participant Storage as Storage
    participant Normalizer as Normalizer
    participant Duplicate as Duplicate Detector
    participant Extractor as Frame Extractor
    participant EDA as Metadata/EDA
    participant Report as Daily Report

    Operator->>Policy: Review source/license
    Policy-->>Queue: Approve candidate

    alt Dry run
        Queue-->>Operator: Validate only, no download
    else Execute mode
        Operator->>Downloader: Run with --execute
        Downloader->>Storage: Download raw source and checksum manifest
        Operator->>Inspector: Inspect media quality and metadata
        Inspector->>Storage: Mark approved / rejected / quarantine
        Storage->>Normalizer: Normalize without altering original
        Normalizer->>Storage: Save processed copy
        Storage->>Duplicate: Group near-duplicates
        Duplicate-->>Storage: Duplicate groups
        Storage->>Extractor: Extract frames and contact sheets
        Extractor->>Storage: Save frames and previews
        Extractor->>EDA: Emit metadata and EDA inputs
        EDA->>Report: Generate summary and daily report
        Report-->>Operator: Deliver review artifacts
    end
```

## Notes
- This diagram is designed for Mermaid-compatible renderers.
- You can save this file as Markdown and preview it in VS Code or GitHub.
- If you want, I can also generate a PNG/SVG version next.
