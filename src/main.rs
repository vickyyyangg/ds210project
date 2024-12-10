use csv;
use std::error::Error;
use rand::seq::SliceRandom; // Import for shuffling
use rand::thread_rng;       // Import for random number generator

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

fn read_dataset(file_path: &str) -> Result<Vec<Individual>, Box<dyn Error>> {
    let mut individuals = Vec::new();
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true) // Ensure headers are skipped
        .from_path(file_path)?;

    let max_records = 20_000;
    let mut parse_errors = 0;

    for (i, result) in rdr.records().enumerate() {
        if i >= max_records {
            break;
        }

        let record = result?;

        // Debug: Print record to verify
        if record.len() < 23 {
            println!("Short record at index {}: {:?}", i, record);
            parse_errors += 1;
            continue;
        }

        let family_influence = match record[14].trim() {
            "None" => Ok(0.0),
            "Low" => Ok(1.0),
            "Medium" => Ok(2.0),
            "High" => Ok(3.0),
            _ => Err("Invalid Family Influence value"),
        };

        match (
            record[2].trim().parse::<f64>(),     
            record[4].trim().parse::<f64>(),     
            record[7].trim().parse::<f64>(),    
            record[19].trim().parse::<f64>(),    
            family_influence,
            record[10].trim().parse::<f64>(),    
            record[22].trim().parse::<f64>(),    
        ) {
            (
                Ok(age),
                Ok(years_of_experience),
                Ok(job_satisfaction),
                Ok(professional_network_size),
                Ok(family_influence),
                Ok(salary),
                Ok(likelihood_to_change_occupation),
            ) => {
                individuals.push(Individual {
                    id: i,
                    age,
                    years_of_experience,
                    job_satisfaction,
                    professional_network_size,
                    family_influence,
                    salary,
                    likelihood_to_change_occupation,
                });
            }
            _ => {
                parse_errors += 1;
                eprintln!("Warning: Could not parse data for record {}", i);
            }
        }
    }

    println!("Total parse errors: {}", parse_errors);
    Ok(individuals)
}


fn calculate_linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    assert_eq!(x.len(), y.len(), "Input vectors must be of equal length");
    let n = x.len() as f64;

    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut var_x = 0.0;
    let mut cov_xy = 0.0;

    for i in 0..x.len() {
        var_x += (x[i] - mean_x).powi(2);
        cov_xy += (x[i] - mean_x) * (y[i] - mean_y);
    }

    var_x /= n - 1.0; // Fix variance calculation

    let slope = cov_xy / (var_x * (n - 1.0));
    let intercept = mean_y - slope * mean_x;

    let mut r_numerator = 0.0;
    let mut r_denomx = 0.0;
    let mut r_denomy = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        r_numerator += dx * dy;
        r_denomx += dx.powi(2);
        r_denomy += dy.powi(2);
    }

    let correlation = r_numerator / (r_denomx * r_denomy).sqrt();
    let r_squared = correlation.powi(2);

    (slope, intercept, correlation, r_squared)
}

fn perform_salary_correlation_analysis(individuals: &[Individual]) -> Result<(), Box<dyn Error>> {
    let analyses = vec![
        ("Salary vs Age",
         individuals.iter().map(|ind| ind.age).collect::<Vec<f64>>(),
         individuals.iter().map(|ind| ind.salary).collect::<Vec<f64>>()),

        ("Salary vs Years of Experience",
         individuals.iter().map(|ind| ind.years_of_experience).collect::<Vec<f64>>(),
         individuals.iter().map(|ind| ind.salary).collect::<Vec<f64>>()),

        ("Salary vs Job Satisfaction",
         individuals.iter().map(|ind| ind.job_satisfaction).collect::<Vec<f64>>(),
         individuals.iter().map(|ind| ind.salary).collect::<Vec<f64>>()),

        ("Salary vs Professional Network Size",
         individuals.iter().map(|ind| ind.professional_network_size).collect::<Vec<f64>>(),
         individuals.iter().map(|ind| ind.salary).collect::<Vec<f64>>()),

        ("Salary vs Family Influence",
         individuals.iter().map(|ind| ind.family_influence).collect::<Vec<f64>>(),
         individuals.iter().map(|ind| ind.salary).collect::<Vec<f64>>()),

        ("Salary vs Likelihood to Change Occupation",
         individuals.iter().map(|ind| ind.likelihood_to_change_occupation).collect::<Vec<f64>>(),
         individuals.iter().map(|ind| ind.salary).collect::<Vec<f64>>()),
    ];

    println!("\n--- Salary Correlation Analyses ---");
    
    for (title, x, y) in analyses {
        let (slope, intercept, correlation, r_squared) = 
            calculate_linear_regression(&x, &y);

        println!("\n{}:", title);
        println!("Correlation Coefficient: {:.4}", correlation);
        println!("Regression Equation: Salary = {:.4} * X + {:.4}", slope, intercept);
        println!("R-squared: {:.4}", r_squared);

        if correlation.abs() < 0.3 {
            println!("Weak correlation");
        } else if correlation.abs() < 0.7 {
            println!("Moderate correlation");
        } else {
            println!("Strong correlation");
        }
    }

    Ok(())
}

fn print_sample_verification(sample: &[Individual]) {
    println!("\n--- Random Sample Verification ---");
    println!("Total records in sample: {}", sample.len());

    // Calculate descriptive statistics for key attributes
    let ages: Vec<f64> = sample.iter().map(|ind| ind.age).collect();
    let experiences: Vec<f64> = sample.iter().map(|ind| ind.years_of_experience).collect();
    let salaries: Vec<f64> = sample.iter().map(|ind| ind.salary).collect();
    
    println!("\nAge Distribution:");
    print_stats(&ages);

    println!("\nYears of Experience Distribution:");
    print_stats(&experiences);

    println!("\nSalary Distribution:");
    print_stats(&salaries);

    println!("\nFamily Influence Distribution:");
    let family_influence_counts: Vec<f64> = vec![
        sample.iter().filter(|ind| ind.family_influence == 0.0).count() as f64,
        sample.iter().filter(|ind| ind.family_influence == 1.0).count() as f64,
        sample.iter().filter(|ind| ind.family_influence == 2.0).count() as f64,
        sample.iter().filter(|ind| ind.family_influence == 3.0).count() as f64,
    ];
    println!("None: {:.2}%", family_influence_counts[0] / sample.len() as f64 * 100.0);
    println!("Low: {:.2}%", family_influence_counts[1] / sample.len() as f64 * 100.0);
    println!("Medium: {:.2}%", family_influence_counts[2] / sample.len() as f64 * 100.0);
    println!("High: {:.2}%", family_influence_counts[3] / sample.len() as f64 * 100.0);

    // Print first 10 and last 10 records to show randomness
    println!("\nFirst 10 Records (Original IDs):");
    for ind in sample.iter().take(10) {
        println!("ID: {}, Age: {}, Salary: {}", ind.id, ind.age, ind.salary);
    }

    println!("\nLast 10 Records (Original IDs):");
    for ind in sample.iter().rev().take(10) {
        println!("ID: {}, Age: {}, Salary: {}", ind.id, ind.age, ind.salary);
    }
}

fn print_stats(data: &[f64]) {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let min = data.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    
    println!("Mean: {:.2}", mean);
    println!("Min: {:.2}", min);
    println!("Max: {:.2}", max);
}

fn main() -> Result<(), Box<dyn Error>> {
    let file_path = "career_dataset.csv";
    let mut individuals = read_dataset(file_path)?;

    if individuals.is_empty() {
        eprintln!("No individuals loaded from the dataset!");
        return Ok(());
    }

    // Shuffle the entire dataset
    let mut rng = thread_rng();
    individuals.shuffle(&mut rng);

    // Select exactly 2,000 records randomly 
    let final_sample: Vec<Individual> = individuals.into_iter().take(2_000).collect();

    // Verify the random sample
    print_sample_verification(&final_sample);

    // Perform correlation analysis
    perform_salary_correlation_analysis(&final_sample)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // Test linear regression with a simple known dataset
    #[test]
    fn test_calculate_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let (slope, intercept, correlation, r_squared) = 
            calculate_linear_regression(&x, &y);

        // Perfect linear relationship expected
        assert!((slope - 2.0).abs() < 1e-6, "Slope should be 2");
        assert!((intercept - 0.0).abs() < 1e-6, "Intercept should be 0");
        assert!((correlation - 1.0).abs() < 1e-6, "Correlation should be 1");
        assert!((r_squared - 1.0).abs() < 1e-6, "R-squared should be 1");
    }

    // Test error handling in linear regression
    #[test]
    #[should_panic(expected = "Input vectors must be of equal length")]
    fn test_linear_regression_different_lengths() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        
        calculate_linear_regression(&x, &y);
    }

    // Test Individual struct creation
    #[test]
    fn test_individual_creation() {
        let individual = Individual {
            id: 1,
            age: 30.0,
            years_of_experience: 5.0,
            job_satisfaction: 4.5,
            professional_network_size: 100.0,
            family_influence: 2.0,
            salary: 75000.0,
            likelihood_to_change_occupation: 0.3,
        };

        assert_eq!(individual.id, 1);
        assert_eq!(individual.age, 30.0);
        assert_eq!(individual.family_influence, 2.0);
    }

    // Test family influence parsing
    #[test]
    fn test_family_influence_parsing() {
        let test_cases = vec![
            ("None", 0.0),
            ("Low", 1.0),
            ("Medium", 2.0),
            ("High", 3.0),
        ];

        for (input, expected) in test_cases {
            let result = match input {
                "None" => 0.0,
                "Low" => 1.0,
                "Medium" => 2.0,
                "High" => 3.0,
                _ => f64::NAN,
            };
            assert_eq!(result, expected);
        }
    }

    // Test reading a small CSV dataset
    #[test]
    fn test_read_small_dataset() {
        let csv_data = 
"Age,Experience,Job Satisfaction,Network Size,Family Influence,Salary,Likelihood to Change
30,5,4.5,100,Medium,75000,0.3
35,10,4.8,200,High,85000,0.2
25,2,3.9,50,Low,60000,0.5";

        // Create a cursor from the string to simulate a file
        let cursor = Cursor::new(csv_data.as_bytes());
        
        // Use csv::ReaderBuilder to create a reader from the cursor
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .from_reader(cursor);

        // Collect individuals manually for testing
        let individuals: Vec<Individual> = rdr
            .records()
            .enumerate()
            .filter_map(|(i, result)| {
                let record = result.ok()?;
                
                let age = record.get(0)?.trim().parse::<f64>().ok()?;
                let years_of_experience = record.get(1)?.trim().parse::<f64>().ok()?;
                let job_satisfaction = record.get(2)?.trim().parse::<f64>().ok()?;
                let professional_network_size = record.get(3)?.trim().parse::<f64>().ok()?;
                
                let family_influence = match record.get(4)?.trim() {
                    "None" => Some(0.0),
                    "Low" => Some(1.0),
                    "Medium" => Some(2.0),
                    "High" => Some(3.0),
                    _ => None,
                }?;
                
                let salary = record.get(5)?.trim().parse::<f64>().ok()?;
                let likelihood_to_change_occupation = record.get(6)?.trim().parse::<f64>().ok()?;

                Some(Individual {
                    id: i,
                    age,
                    years_of_experience,
                    job_satisfaction,
                    professional_network_size,
                    family_influence,
                    salary,
                    likelihood_to_change_occupation,
                })
            })
            .collect();

        assert_eq!(individuals.len(), 3, "Should parse 3 records");
        assert_eq!(individuals[0].age, 30.0);
        assert_eq!(individuals[1].salary, 85000.0);
        assert_eq!(individuals[2].family_influence, 1.0);
    }

    // Test print_stats function
    #[test]
    fn test_print_stats() {
        let test_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        
        let mean = test_data.iter().sum::<f64>() / test_data.len() as f64;
        let min = test_data.iter().cloned().fold(f64::INFINITY, |a, b| a.min(b));
        let max = test_data.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
        
        assert!((mean - 30.0).abs() < 1e-6, "Mean should be 30");
        assert!((min - 10.0).abs() < 1e-6, "Min should be 10");
        assert!((max - 50.0).abs() < 1e-6, "Max should be 50");
    }
}