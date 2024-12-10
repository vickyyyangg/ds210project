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