extern crate ndarray;

use ndarray::prelude::*;
use ndarray_linalg::*;

struct LinearRegression {
    alpha: f64,
}

impl LinearRegression {
    fn train(X: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        X + y
    }
}


fn main() {
    let a = arr2(&[[1, 2],
                   [2, 3]]);

    let b = a.inv();

    println!("{:?}", b);
}

fn f(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    a * b
}

