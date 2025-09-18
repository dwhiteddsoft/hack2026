use criterion::{black_box, criterion_group, criterion_main, Criterion};
use uocvr::utils::math_utils;

fn bench_iou_calculation(c: &mut Criterion) {
    let box1 = [0.0, 0.0, 10.0, 10.0];
    let box2 = [5.0, 5.0, 15.0, 15.0];

    c.bench_function("iou_calculation", |b| {
        b.iter(|| math_utils::calculate_iou(black_box(&box1), black_box(&box2)))
    });
}

fn bench_sigmoid(c: &mut Criterion) {
    c.bench_function("sigmoid", |b| {
        b.iter(|| math_utils::sigmoid(black_box(0.5)))
    });
}

fn bench_softmax(c: &mut Criterion) {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    
    c.bench_function("softmax", |b| {
        b.iter(|| {
            let mut vals = values.clone();
            math_utils::softmax(black_box(&mut vals))
        })
    });
}

criterion_group!(benches, bench_iou_calculation, bench_sigmoid, bench_softmax);
criterion_main!(benches);