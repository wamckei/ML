print("🚀 Starting training... Monitor: watch nvidia-smi -l 1")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator 
)

trainer.train()

# Save final model
trainer.save_model("./my-python-code-llm-final_qwen")
print("✅ Training completed! Model saved.")
