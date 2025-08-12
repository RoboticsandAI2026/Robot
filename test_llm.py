import torch
import json
import os
from obstacle_llm import SimpleTokenizer, SimpleObstacleLLM


class ModelTester:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tokenizer = None

    def load_model(self, model_dir='saved_models'):
        # Load the paths to the latest model and tokenizer
        paths_file = os.path.join(model_dir, 'latest_model_paths.json')
        if not os.path.exists(paths_file):
            raise FileNotFoundError("No saved model found. Please train the model first.")

        with open(paths_file, 'r') as f:
            paths = json.load(f)

        # Load tokenizer
        with open(paths['tokenizer_path'], 'r') as f:
            vocab = json.load(f)
        self.tokenizer = SimpleTokenizer()
        self.tokenizer.vocab = vocab
        self.tokenizer.id2word = {v: k for k, v in vocab.items()}

        # Load model
        checkpoint = torch.load(paths['model_path'], map_location=self.device)
        self.model = SimpleObstacleLLM(checkpoint['vocab_size'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print("Model and tokenizer loaded successfully!")

    def generate_response(self, input_text):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded first!")

        with torch.no_grad():
            tokens = self.tokenizer.encode(input_text)
            tokens = [self.tokenizer.vocab['<START>']] + tokens
            input_tensor = torch.LongTensor([tokens]).to(self.device)
            output_tokens = self.model.generate(input_tensor)
            return self.tokenizer.decode(output_tokens[0].tolist())


def main():
    # Initialize tester
    tester = ModelTester()

    try:
        # Load the model
        tester.load_model()

        # Predefined test questions
        test_questions = [
            "what is the purpose of the water bottle",
            "describe the shoe",
            "tell me about the cap",
            "identify the dust pan",
            "what is a football"
        ]

        while True:
            print("\nSelect an option:")
            print("1. Run predefined test questions")
            print("2. Enter your own question")
            print("3. Exit")

            choice = input("Enter your choice (1-3): ")

            if choice == '1':
                print("\nRunning predefined test questions:")
                for question in test_questions:
                    response = tester.generate_response(question)
                    print(f"\nQ: {question}")
                    print(f"A: {response}")

            elif choice == '2':
                question = input("\nEnter your question: ")
                response = tester.generate_response(question)
                print(f"\nQ: {question}")
                print(f"A: {response}")

            elif choice == '3':
                print("Goodbye!")
                break

            else:
                print("Invalid choice. Please try again.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()