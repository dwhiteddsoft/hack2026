# Project Context for AI Assistants

## Project Overview

### Basic Information
- **Project Name**: `data-fusion-rs`
- **Version**: 0.3.0 (pre-1.0, API may change)
- **Primary Language**: Rust (edition 2021)
- **License**: MIT OR Apache-2.0
- **Repository**: https://github.com/username/data-fusion-rs

### Purpose & Mission
A high-performance Rust library for real-time data processing and analysis. Designed for applications that need to process large datasets with minimal memory overhead and maximum throughput. The library provides zero-copy data transformations, streaming processing capabilities, and ergonomic APIs for data scientists and systems engineers.

### Target Users
- Systems engineers building data pipelines
- Data scientists needing performance-critical processing
- Backend developers handling real-time analytics
- Anyone needing memory-efficient data transformations

## Technical Architecture

### Core Technologies
- **Language**: Rust 1.70+ (MSRV: 1.70)
- **Async Runtime**: Tokio (for I/O-bound operations)
- **Serialization**: Serde with custom optimizations
- **Parallelization**: Rayon for CPU-bound work
- **Error Handling**: Custom error types with `thiserror`
- **Testing**: Standard `cargo test` + `proptest` for property-based testing
- **Benchmarking**: Criterion.rs for performance measurement

### Architecture Patterns
- **Zero-Copy Design**: Extensive use of borrowing, `Cow<T>`, and view types
- **Builder Pattern**: For complex configuration objects
- **Type-State Pattern**: Compile-time state validation
- **Plugin Architecture**: Extensible processing pipeline via traits
- **Memory Pool Management**: Custom allocators for high-frequency operations

### Key Modules
```
src/
├── lib.rs              # Public API and re-exports
├── core/               # Core data structures and algorithms
│   ├── dataset.rs      # Main Dataset struct
│   ├── column.rs       # Column implementations
│   └── transforms.rs   # Data transformation engine
├── io/                 # Input/output operations
│   ├── csv.rs          # CSV reading/writing
│   ├── parquet.rs      # Parquet format support
│   └── streaming.rs    # Streaming I/O primitives
├── compute/            # Computation engine
│   ├── aggregations.rs # Sum, mean, group-by operations
│   ├── filters.rs      # Data filtering logic
│   └── joins.rs        # Join algorithms
└── error.rs           # Centralized error handling
```

## Business Logic & Domain Knowledge

### Key Concepts
- **Dataset**: Immutable, columnar data structure (think Apache Arrow but simpler)
- **Transform**: Zero-copy operation that produces a view of existing data
- **Pipeline**: Chain of transforms that can be executed lazily
- **Chunk**: Fixed-size batch of rows for memory-efficient processing
- **Schema**: Type information for datasets with compile-time validation

### Performance Requirements
- **Memory**: Target <1GB for processing 100M rows
- **Throughput**: >1M rows/second on modern hardware
- **Latency**: <10ms for simple transforms on 10K rows
- **Concurrency**: Full thread-safety with `Send + Sync`

### Data Types Supported
- Primitive types: i32, i64, f32, f64, bool, String
- Temporal: DateTime, Date, Time (chrono integration)
- Optional: All types wrapped in Option<T>
- Collections: Vec<T>, arrays (future: nested structures)

## Development Workflow

### Current Development Phase
We are in **pre-1.0 stabilization phase**:
- API design is mostly stable but may have breaking changes
- Focus on performance optimization and memory usage
- Adding comprehensive benchmarks and real-world testing
- Documentation and example improvements

### Code Quality Standards
- **Test Coverage**: Maintain >90% coverage
- **Documentation**: All public APIs must have examples
- **Performance**: Benchmark all public operations
- **Safety**: Zero unsafe code in public API surface
- **Compatibility**: Support stable Rust + 3 previous versions

### Git Workflow
- `main` branch for stable releases
- `develop` branch for active development
- Feature branches: `feature/description`
- Releases tagged as `v0.3.0`, etc.

## Technical Constraints & Decisions

### Architectural Decision Records (ADRs)
1. **ADR-001**: Choose columnar storage over row-based for cache efficiency
2. **ADR-002**: Use `Cow<T>` extensively to balance zero-copy with flexibility
3. **ADR-003**: Custom error types over `anyhow` for better API ergonomics
4. **ADR-004**: Async I/O but sync compute to avoid runtime overhead
5. **ADR-005**: Plugin system via traits rather than dynamic dispatch for performance

### Known Limitations
- No nested data structures yet (planned for v0.4)
- Limited string processing optimizations
- No distributed computing support (out of scope)
- Memory usage spikes during joins (optimization target)

### Dependencies Strategy
- **Minimal dependencies**: Only add deps that provide significant value
- **No-std compatibility**: Core types should work in `no-std` environments
- **Version pinning**: Pin major versions to avoid breaking changes

## AI Assistant Guidelines

### Context Awareness
When working on this project, remember:
- Performance implications of every API decision
- Memory layout considerations for cache efficiency
- The zero-copy philosophy throughout the design
- Backwards compatibility concerns (pre-1.0 but stability matters)

### Domain-Specific Language
- "Column" = vertical slice of data (all values of one field)
- "Row" = horizontal slice of data (one record across all fields)
- "Transform" = operation that produces a view without copying data
- "Materialization" = when we actually compute and store results
- "Schema" = type and structure information about the data

### Common Patterns in This Codebase
- Heavy use of iterators and `Iterator` trait implementations
- `Result<T, DataFusionError>` for all fallible operations
- Builder pattern: `DatasetBuilder::new().add_column().build()`
- Lazy evaluation: transforms return closures that execute on demand
- Type erasure: `Box<dyn DataType>` for runtime type handling

### Testing Philosophy
- **Unit tests**: Test individual functions in isolation
- **Integration tests**: Test complete workflows end-to-end
- **Property tests**: Use `proptest` to verify invariants
- **Benchmarks**: Measure performance impact of all changes
- **Examples**: Every public API has a working example

### Performance Priorities
1. Memory efficiency (minimize allocations and copies)
2. Cache locality (columnar layout, predictable access patterns)
3. Parallelization opportunities (embarrassingly parallel operations)
4. Lazy evaluation (don't compute until necessary)
5. SIMD optimization potential (future enhancement)

## Current Sprint Goals
- [ ] Implement advanced aggregation functions (percentiles, variance)
- [ ] Optimize join algorithms for large datasets
- [ ] Add comprehensive error recovery mechanisms
- [ ] Improve documentation with real-world examples
- [ ] Performance audit with profiling tools