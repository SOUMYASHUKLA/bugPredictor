from predict import BugPredictionService

# Initialize predictor
predictor = BugPredictionService('checkpoints/model_epoch_2.pt')

# Make prediction
result = predictor.predict(
    commit_message="Fix null pointer exception",
    code_changes="if (user != null) { user.process(); }"
)

print(f"Risk Level: {result['risk_level']}")
print(f"Bug Probability: {result['probability']:.2%}")
print(f"Explanation: {result['explanation']}")