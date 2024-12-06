use std::io;
use std::io::Read;
use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};
use rand::{random, Rng, thread_rng};
use std::time::Instant;
use dirs;

//TODO: allow for 'personality'/system message prompts

fn main() {

    println!("Which model? 1 (fast) or 2 (slow)"); // 
    let mut model = String::new();
    io::stdin()
        .read_line(&mut model)
        .expect("Failed to read line");
    let model = model.trim();

    let model_options = ModelOptions {
        n_gpu_layers: if model == "1" {35} else {8},
        ..Default::default()
    };

    let documents_path = dirs::home_dir().expect("Failed to get home directory").join("Documents").join("LLM");
    let llama = match model {
        // TODO: Allow for the model to be stored anywhere, as long as it is in the same folder as the executable
        // TODO: Allow for more modular choice of LLM model
        "1" => LLama::new(documents_path.join("mistral-7b-instruct-v0.1.Q5_K_S.gguf"), &model_options).unwrap(),
        "2" => LLama::new(documents_path.join("mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"), &model_options).unwrap(),
        _ => panic!("Invalid input")
    };

    let mut buffer;
    let mut history = String::new();
    loop {
        let predict_options = PredictOptions {
            token_callback: Some(Box::new(|token| {
                print!("{token}");
                true
            })),
            tokens:99999,
            stop_prompts:vec![],
            seed:random::<i32>(),
            ..Default::default()
        };
        println!("\nYour turn:");

        buffer = String::new();
        io::stdin()
            .read_line(&mut buffer)
            .expect("Failed to read line");

        println!("LLM's turn:");
        let timer = Instant::now();
        let response = llama.predict(format!("{}<s>[INST] {} [/INST]", history, buffer), predict_options).unwrap();
        println!("\nResponse took {} seconds", timer.elapsed().as_secs_f32());
        history.push_str(&format!("<s>[INST] {} [/INST] {} </s>\n", buffer, response.0));
    }
}