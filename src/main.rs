use csv;
use std::error::Error;
use rand::seq::SliceRandom; 
use rand::thread_rng;      

#[derive(Debug, Clone)]
struct Individual {
    #[allow(dead_code)]
    id: usize,
    age: f64,
    years_of_experience: f64,
    job_satisfaction: f64,
    professional_network_size: f64,
    family_influence: f64, // Ordinal encoding: Low → 1, Medium → 2, High → 3
    salary: f64,
    likelihood_to_change_occupation: f64,
}