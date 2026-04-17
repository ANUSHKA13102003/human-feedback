import json
import random
import pandas as pd
from typing import List, Dict, Tuple

class DatasetGenerator:
    """Generate synthetic prompts and multiple responses for RLHF training."""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.prompts_templates = [
            "Explain {topic} in simple terms.",
            "What are the key benefits of {topic}?",
            "How can I learn {topic}?",
            "What is {topic} used for?",
            "Compare {topic} with alternatives.",
            "List the steps to {topic}.",
            "Why is {topic} important?",
            "What are common mistakes with {topic}?",
            "How does {topic} work?",
            "What tools can help with {topic}?",
        ]
        
        self.topics = [
            "machine learning",
            "Python programming",
            "data science",
            "web development",
            "cloud computing",
            "cybersecurity",
            "artificial intelligence",
            "database design",
            "REST APIs",
            "containerization with Docker",
        ]
        
        self.response_templates = {
            "good": [
                "A comprehensive explanation: {topic} is fundamental in modern tech. It involves {detail1}, {detail2}, and {detail3}. Key applications include {app1} and {app2}. Best practices recommend focusing on {practice1} and {practice2}.",
                "{topic} can be understood as: {detail1} combined with {detail2}. The main advantages are {app1}, {app2}, and improved {practice1}. Start by learning {detail1}, then move to {detail2}.",
                "Here's a detailed breakdown: {topic} encompasses {detail1}, {detail2}, and {detail3}. It's essential for {app1} and {app2}. Recommended approaches include {practice1} and {practice2}.",
            ],
            "moderate": [
                "{topic} is about {detail1} and {detail2}. It has uses in {app1}. You should know about {practice1}.",
                "To understand {topic}, focus on {detail1}. It relates to {detail2} and {app1}.",
                "{topic} involves {detail1}. Important aspects: {detail2}, {app1}.",
            ],
            "poor": [
                "{topic} is good.",
                "Yes, {topic} is important.",
                "I know about {topic}. It exists.",
                "{topic} is a thing.",
            ]
        }
        
        self.details = [
            "pattern recognition",
            "statistical analysis",
            "optimization",
            "data manipulation",
            "system design",
            "algorithm efficiency",
            "scalability",
            "security protocols",
            "user experience",
            "performance tuning",
        ]
        
        self.applications = [
            "predictive analytics",
            "automation",
            "decision making",
            "risk assessment",
            "efficiency improvement",
            "quality assurance",
            "innovation",
            "cost reduction",
        ]
        
        self.practices = [
            "continuous learning",
            "code review",
            "testing",
            "documentation",
            "collaboration",
            "monitoring",
            "iteration",
        ]
    
    def generate_prompt(self) -> str:
        """Generate a single prompt."""
        template = random.choice(self.prompts_templates)
        topic = random.choice(self.topics)
        return template.format(topic=topic)
    
    def generate_response(self, quality: str) -> str:
        """Generate a response of specific quality."""
        template = random.choice(self.response_templates[quality])
        topic = random.choice(self.topics)
        detail1 = random.choice(self.details)
        detail2 = random.choice(self.details)
        detail3 = random.choice(self.details)
        app1 = random.choice(self.applications)
        app2 = random.choice(self.applications)
        practice1 = random.choice(self.practices)
        practice2 = random.choice(self.practices)
        
        return template.format(
            topic=topic,
            detail1=detail1,
            detail2=detail2,
            detail3=detail3,
            app1=app1,
            app2=app2,
            practice1=practice1,
            practice2=practice2,
        )
    
    def generate_dataset(self, n_prompts: int = 100, responses_per_prompt: int = 4) -> pd.DataFrame:
        """Generate a dataset with prompts and multiple responses."""
        data = []
        
        for prompt_idx in range(n_prompts):
            prompt = self.generate_prompt()
            
            # Generate responses with different quality levels
            responses = []
            quality_scores = []
            
            for resp_idx in range(responses_per_prompt):
                if resp_idx == 0:
                    quality = "good"
                    score = random.uniform(0.8, 1.0)
                elif resp_idx == 1:
                    quality = "good"
                    score = random.uniform(0.75, 0.95)
                elif resp_idx == 2:
                    quality = "moderate"
                    score = random.uniform(0.5, 0.7)
                else:
                    quality = "poor"
                    score = random.uniform(0.1, 0.4)
                
                response = self.generate_response(quality)
                responses.append(response)
                quality_scores.append(round(score, 2))
            
            # Create records for each response
            for resp_idx, (response, score) in enumerate(zip(responses, quality_scores)):
                data.append({
                    "prompt_id": prompt_idx,
                    "prompt": prompt,
                    "response_id": resp_idx,
                    "response": response,
                    "quality_score": score,
                })
        
        df = pd.DataFrame(data)
        return df


def main():
    """Generate and save the dataset."""
    generator = DatasetGenerator(seed=42)
    
    # Generate dataset
    print("Generating RLHF dataset...")
    df = generator.generate_dataset(n_prompts=100, responses_per_prompt=4)
    
    # Save raw dataset
    df.to_csv("data/raw_dataset.csv", index=False)
    print(f"Saved raw dataset: data/raw_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Generate summary statistics
    print(f"\nQuality score statistics:")
    print(df["quality_score"].describe())


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()