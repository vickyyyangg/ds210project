use csv;
use std::error::Error;

#[derive(Debug, Clone)]
struct Individual {
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
    let mut rdr = csv::Reader::from_path(file_path)?;

    for (i, result) in rdr.records().enumerate() {
        let record = result?;

        // Ordinal encoding for Family Influence
        let family_influence = match record[14].trim() { // Assuming index 15 for Family Influence
            "Low" => Ok(1.0),
            "Medium" => Ok(2.0),
            "High" => Ok(3.0),
            _ => Err("Invalid Family Influence value"),
        };

        // Parse other fields
        match (
            record[2].trim().parse::<f64>(),     // Age (index 2)
            record[4].trim().parse::<f64>(),     // Years of Experience (index 4)
            record[7].trim().parse::<f64>(),     // Job Satisfaction (index 7)
            record[19].trim().parse::<f64>(),    // Professional Network Size (index 19)
            family_influence,
            record[10].trim().parse::<f64>(),    // Salary (index 11)
            record[16].trim().parse::<f64>(),    // Likelihood to Change Occupation (index 17)
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
                eprintln!("Warning: Could not parse data for record {}", i);
            }
        }
    }
    Ok(individuals)
}
