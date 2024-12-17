import json
import os
from swarm import Swarm, Agent, Response
from pathlib import Path
import os
from dotenv import load_dotenv
import concurrent.futures

# load_dotenv()
# os.environ['OPENAI_API_KEY']=""

OUTPUT_DIR = "agent_learnings"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

def save_prompt_to_txt(agent_name: str, prompt: str):
    """Save an agent's generated prompt to a text file"""
    filename = f"{OUTPUT_DIR}/{agent_name}_prompt.txt"
    with open(filename, 'w') as f:
        f.write(prompt)
    return f"Saved prompt to {filename}"

def read_sample_output():
    """Read the sample output file"""
    with open("sample_output.txt", 'r') as f:
        return f.read()

# Create a single Swarm client instance
client = Swarm()

# Create the agents first
grammar_agent = Agent(
    name="GrammarAgent",
    model="gpt-4o-mini",
    instructions="Analyze the grammar patterns in the provided text and generate a prompt that would produce text with similar grammatical characteristics."
)

tone_agent = Agent(
    name="ToneAgent",
    model="gpt-4o-mini",
    instructions="Analyze the tone and style of the provided text and generate a prompt that would produce text with a similar tone."
)

readability_agent = Agent(
    name="ReadabilityAgent",
    model="gpt-4o-mini",
    instructions="Analyze the readability level and structure of the provided text and generate a prompt that would produce similarly readable text."
)

aggregator_agent = Agent(
    name="AggregatorAgent",
    model="gpt-4o-mini",
    instructions="Combine the learnings from other agents to create a comprehensive prompt that captures all aspects of the desired text generation."
)

# Add new agents after the existing ones
reading_score_agent = Agent(
    name="ReadingScoreAgent",
    model="gpt-4o-mini",
    instructions="Analyze the text's complexity and determine its suitability for different age groups using metrics like Flesch-Kincaid, SMOG, or similar readability indexes."
)

formatting_agent = Agent(
    name="FormattingAgent",
    model="gpt-4o-mini",
    instructions="Analyze the text's formatting patterns including paragraph structure, spacing, use of headings, lists, and other structural elements."
)

lexical_density_agent = Agent(
    name="LexicalDensityAgent",
    model="gpt-4o-mini",
    instructions="Calculate and analyze the ratio of content words (nouns, verbs, adjectives, adverbs) to total words. Evaluate the text's information density and complexity based on lexical patterns."
)

emotional_intensity_agent = Agent(
    name="EmotionalIntensityAgent",
    model="gpt-4o-mini",
    instructions="Measure the strength and variation of emotional expressions in the text. Analyze emotional peaks, patterns, and the overall emotional journey."
)

# Add new agents for advanced analysis
sentiment_agent = Agent(
    name="SentimentAgent",
    model="gpt-4o-mini",
    instructions="Analyze the text's overall sentiment, emotional valence, and identify dominant emotional patterns and their intensity levels."
)

complexity_measuring_agent = Agent(
    name="ComplexityMeasuringAgent",
    model="gpt-4o-mini",
    instructions="Analyze sentence length, word complexity, syntactic structures, and nested clauses to measure overall textual complexity."
)

def analyze_with_agent(agent, analysis_type):
    """Generic function to analyze text with a given agent"""
    text = read_sample_output()
    analysis_prompt = (
        f"Analyze the following text for its {analysis_type}. "
        "Based on your analysis, generate a system prompt that would help "
        f"produce text with similar {analysis_type} characteristics:\n\n"
        f"{text}"
    )
    
    response = client.run(
        agent=agent,
        messages=[{"role": "user", "content": analysis_prompt}]
    )
    return save_prompt_to_txt(agent.name.lower().replace('agent', ''), response.messages[0].get('content'))

def analyze_grammar():
    return analyze_with_agent(grammar_agent, "grammatical patterns, sentence structures, and writing conventions")

def analyze_tone():
    return analyze_with_agent(tone_agent, "tone, style, formality level, and overall voice")

def analyze_readability():
    return analyze_with_agent(readability_agent, "readability level, complexity, technical depth, and information structure")

# Add new analysis functions
def analyze_reading_score():
    return analyze_with_agent(reading_score_agent, 
        "reading level and age group suitability, considering factors like: "
        "vocabulary complexity, sentence length, concept difficulty, and "
        "overall accessibility for different age groups and education levels")

def analyze_formatting():
    return analyze_with_agent(formatting_agent,
        "formatting characteristics including: paragraph structure, "
        "use of whitespace, heading hierarchy, list formatting, "
        "text emphasis patterns, and overall document organization")

def analyze_emotional_intensity():
    return analyze_with_agent(emotional_intensity_agent,
        "emotional intensity patterns including: strength of expressions, "
        "emotional variation, intensity peaks and valleys, and "
        "overall emotional arc of the content")
# Add corresponding analysis functions
def analyze_sentiment():
    return analyze_with_agent(sentiment_agent,
        "sentiment patterns including: emotional valence, "
        "sentiment progression, emotional undertones, and "
        "overall emotional impact of the text")

def analyze_lexical_density():
    return analyze_with_agent(lexical_density_agent,
        "lexical density characteristics including: content word ratio, "
        "vocabulary richness, information density patterns, and "
        "distribution of word types throughout the text")


def analyze_complexity():
    return analyze_with_agent(complexity_measuring_agent,
        "complexity metrics including: sentence length patterns, "
        "word complexity levels, syntactic structure variety, and "
        "overall textual sophistication")

def aggregate_prompts():
    """Read all prompt files and generate a final combined prompt"""
    prompts = {}
    for filename in os.listdir(OUTPUT_DIR):
        if filename.endswith("_prompt.txt"):
            agent_name = filename.replace("_prompt.txt", "")
            with open(f"{OUTPUT_DIR}/{filename}", 'r') as f:
                prompts[agent_name] = f.read()
    
    aggregation_prompt = (
        "You are tasked with creating a single, comprehensive system prompt based on "
        "the following specialized analyses. Each analysis focuses on a different aspect "
        "of the text. Create a unified prompt that captures all these characteristics "
        "in a clear, concise way that can be used to generate similar text.\n\n"
        
        f"Grammar Analysis:\n{prompts.get('grammar', 'Not available')}\n\n"
        f"Tone Analysis:\n{prompts.get('tone', 'Not available')}\n\n"
        f"Readability Analysis:\n{prompts.get('readability', 'Not available')}\n\n"
        f"Reading Score Analysis:\n{prompts.get('readingscore', 'Not available')}\n\n"
        f"Formatting Analysis:\n{prompts.get('formatting', 'Not available')}\n\n"
        f"Sentiment Analysis:\n{prompts.get('sentiment', 'Not available')}\n\n"
        f"Lexical Density Analysis:\n{prompts.get('lexicaldensity', 'Not available')}\n\n"
        f"Emotional Intensity Analysis:\n{prompts.get('emotionalintensity', 'Not available')}\n\n"
        f"Complexity Analysis:\n{prompts.get('complexity', 'Not available')}\n\n"
        
        "Generate a single, cohesive system prompt that incorporates all these aspects. "
        "The prompt should be clear, actionable, and suitable for use in other applications. "
        "Ensure the final prompt addresses grammar, tone, readability, formatting, emotional aspects, "
        "and technical complexity in a balanced way."
    )
    
    # Use the aggregator agent to generate the final prompt
    response = client.run(
        agent=aggregator_agent,
        messages=[{"role": "user", "content": aggregation_prompt}]
    )
    
    return save_prompt_to_txt("final", response.messages[0].get('content'))

def main():
    # Update analysis functions list to include new agents
    analysis_functions = [
        analyze_grammar, 
        analyze_tone, 
        analyze_readability,
        analyze_reading_score,
        analyze_formatting,
        analyze_sentiment,
        analyze_lexical_density,
        analyze_emotional_intensity,
        analyze_complexity
    ]
    
    print("Starting parallel analysis...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Updated max_workers
        future_to_func = {executor.submit(func): func for func in analysis_functions}
        
        for future in concurrent.futures.as_completed(future_to_func):
            func = future_to_func[future]
            try:
                result = future.result(timeout=60)
                print(f"{func.__name__} completed analysis")
            except concurrent.futures.TimeoutError:
                print(f"{func.__name__} timed out")
            except Exception as e:
                print(f"{func.__name__} failed with error: {str(e)}")
    
    print("All analyses complete. Aggregating results...")
    final_result = aggregate_prompts()
    print("Final prompt generated and saved")

if __name__ == "__main__":
    main()
