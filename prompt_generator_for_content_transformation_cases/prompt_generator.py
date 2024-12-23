import logging
from swarm import Swarm, Agent, Response
import os
from concurrent.futures import ThreadPoolExecutor
import asyncio
import os
from dotenv import load_dotenv
import concurrent.futures

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()
logger.info("Environment variables loaded")

# Initialize Swarm client
client = Swarm()
logger.info("Swarm client initialized")

OUTPUT_DIR = "agent_learnings"
os.makedirs(OUTPUT_DIR, exist_ok=True)
logger.info(f"Output directory ensured at: {OUTPUT_DIR}")

def read_files():
    """Read input and output files"""
    logger.info("Reading input and output files")
    try:
        with open("input.txt", "r") as f:
            input_text = f.read()
        with open("output.txt", "r") as f:
            output_text = f.read()
        logger.info("Successfully read input and output files")
        return input_text, output_text
    except FileNotFoundError as e:
        logger.error(f"Failed to read files: {str(e)}")
        raise

def write_learning(agent_name, learning):
    """Write agent's learning to file"""
    filename = os.path.join(OUTPUT_DIR, f"{agent_name}_learning.txt")
    logger.info(f"Writing learning for agent '{agent_name}' to {filename}")
    try:
        with open(filename, "w") as f:
            f.write(learning)
        logger.info(f"Successfully wrote learning for agent '{agent_name}'")
        return filename
    except Exception as e:
        logger.error(f"Failed to write learning for agent '{agent_name}': {str(e)}")
        raise

def execute_prompt(agent, analysis_prompt):
    """Common function to execute prompt and save results"""
    try:
        logger.info(f"Executing prompt for agent: {agent.name}")
        response = client.run(
            agent=agent,
            messages=[{"role": "user", "content": analysis_prompt}]
        )
        
        # Extract content from response
        content = response.messages[-1].get('content', 'No content available')
        if not content or content == 'No content available':
            logger.warning(f"No content received from {agent.name}")
        
        # Save to file and return
        filename = write_learning(
            agent.name.lower().replace('agent', ''), 
            content
        )
        logger.info(f"Successfully executed prompt for {agent.name} and saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error executing prompt for {agent.name}: {str(e)}")
        raise

def create_analysis_prompt(input_text, output_text, analysis_points):
    """Create standardized analysis prompt"""
    return f"""Analyze the following text transformation:
    
    INPUT TEXT:
    {input_text}
    
    OUTPUT TEXT:
    {output_text}
    
    Provide detailed analysis focusing on:
    {analysis_points}
    
    Format your response as a structured analysis with clear sections for each point."""

async def analyze_grammar(input_text, output_text):
    """Grammar analysis function"""
    analysis_points = """
    1. Sentence structure changes
    2. Grammar rule applications
    3. Tense and voice modifications
    4. Syntactic pattern transformations"""
    
    agent = Agent(
        name="GrammarAgent",
        instructions="Analyze grammar transformations between texts",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_tone(input_text, output_text):
    """Tone analysis function"""
    analysis_points = """
    1. Formality level changes
    2. Voice consistency
    3. Style modifications
    4. Tone pattern shifts"""
    
    agent = Agent(
        name="ToneAgent",
        instructions="Analyze tone and style transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_readability(input_text, output_text):
    """Readability analysis function"""
    analysis_points = """
    1. Sentence complexity changes
    2. Text flow modifications
    3. Readability improvements
    4. Structure clarity changes"""
    
    agent = Agent(
        name="ReadabilityAgent",
        instructions="Analyze readability transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_reading_score(input_text, output_text):
    """Reading score analysis function"""
    analysis_points = """
    1. Reading level shifts
    2. Complexity metrics (Flesch-Kincaid, SMOG)
    3. Age group suitability changes
    4. Comprehension level adjustments"""
    
    agent = Agent(
        name="ReadingScoreAgent",
        instructions="Analyze reading level transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_formatting(input_text, output_text):
    """Formatting analysis function"""
    analysis_points = """
    1. Paragraph structure changes
    2. Spacing and layout modifications
    3. List and heading formatting
    4. Visual organization patterns"""
    
    agent = Agent(
        name="FormattingAgent",
        instructions="Analyze formatting transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_lexical_density(input_text, output_text):
    """Lexical density analysis function"""
    analysis_points = """
    1. Content word ratio changes
    2. Information density shifts
    3. Vocabulary complexity changes
    4. Word usage patterns"""
    
    agent = Agent(
        name="LexicalDensityAgent",
        instructions="Analyze lexical density transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_emotional_intensity(input_text, output_text):
    """Emotional intensity analysis function"""
    analysis_points = """
    1. Emotional peak changes
    2. Intensity pattern shifts
    3. Emotional journey mapping
    4. Affective content changes"""
    
    agent = Agent(
        name="EmotionalIntensityAgent",
        instructions="Analyze emotional intensity transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_sentiment(input_text, output_text):
    """Sentiment analysis function"""
    analysis_points = """
    1. Sentiment polarity changes
    2. Emotional valence shifts
    3. Tone consistency patterns
    4. Mood transformation mapping"""
    
    agent = Agent(
        name="SentimentAgent",
        instructions="Analyze sentiment transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_complexity(input_text, output_text):
    """Complexity analysis function"""
    analysis_points = """
    1. Sentence length variations
    2. Structural complexity changes
    3. Cognitive load modifications
    4. Comprehension difficulty shifts"""
    
    agent = Agent(
        name="ComplexityAgent",
        instructions="Analyze complexity transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

async def analyze_text_size(input_text, output_text):
    """Text size analysis function"""
    analysis_points = """
    1. Character count changes
    2. Word count variations
    3. Content expansion/reduction patterns
    4. Section-specific size changes
    5. Overall length transformation"""
    
    agent = Agent(
        name="TextSizeAgent",
        instructions="Analyze size transformations",
        model="gpt-4o-mini"
    )
    
    prompt = create_analysis_prompt(input_text, output_text, analysis_points)
    return await asyncio.to_thread(execute_prompt, agent, prompt)

def aggregate_results(results):
    """Aggregate all agent learnings and generate final transformation prompt"""
    try:
        print("\nStarting results aggregation...")
        
        # Read all learning files
        learnings = {}
        for filename in os.listdir(OUTPUT_DIR):
            if filename.endswith('_learning.txt'):
                with open(os.path.join(OUTPUT_DIR, filename), 'r') as f:
                    agent_name = filename.replace('_learning.txt', '')
                    learnings[agent_name] = f.read()
        
        # Create aggregation prompt
        aggregation_prompt = f"""As an expert prompt engineer, analyze the following transformation analysis reports and create a comprehensive prompt that can reproduce similar transformations.

        Analysis Reports:
        {'-' * 80}
        """
        
        for agent_name, learning in learnings.items():
            aggregation_prompt += f"\n{agent_name} Analysis:\n{learning}\n{'-' * 80}\n"
        
        aggregation_prompt += """
        Based on these analyses, create a detailed prompt that:
        1. Captures all identified transformation patterns
        2. Provides clear instructions for text transformation
        3. Includes specific rules for:
           - Grammar and structure changes
           - Style and tone modifications
           - Formatting requirements
           - Content density adjustments
           - Size and length transformations
        4. Can be used to consistently reproduce similar transformations
        
        Format the prompt as a clear, structured set of instructions that a language model can follow.
        """
        
        # Create aggregator agent
        aggregator = Agent(
            name="FinalAggregator",
            instructions="Create comprehensive transformation prompt from analysis reports",
            model="gpt-4o-mini"
        )
        
        # Generate final prompt
        print("Generating final transformation prompt...")
        response = client.run(
            agent=aggregator,
            messages=[{"role": "user", "content": aggregation_prompt}]
        )
        
        # Extract and save final prompt
        final_prompt = response.messages[-1].get('content', 'No content available')
        final_prompt_path = write_learning('final_transformation_prompt', final_prompt)
        
        print(f"\nFinal transformation prompt generated and saved to: {final_prompt_path}")
        
        return final_prompt
        
    except Exception as e:
        print(f"Error in aggregating results: {str(e)}")
        raise

async def main():
    try:
        logger.info("Starting analysis pipeline")
        input_text, output_text = read_files()
        
        analysis_functions = [
            analyze_grammar,
            analyze_tone,
            analyze_readability,
            analyze_reading_score,
            analyze_formatting,
            analyze_lexical_density,
            analyze_emotional_intensity,
            analyze_sentiment,
            analyze_complexity,
            analyze_text_size
        ]
        
        logger.info(f"Launching {len(analysis_functions)} parallel analysis tasks")
        
        # Execute analyses in parallel
        tasks = [
            func(input_text, output_text)
            for func in analysis_functions
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check for exceptions in results
        failed_tasks = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_tasks += 1
                logger.error(f"Task {analysis_functions[i].__name__} failed with error: {result}")
        
        if failed_tasks:
            logger.warning(f"{failed_tasks} out of {len(results)} tasks failed")
        else:
            logger.info("All analysis tasks completed successfully")
        
        # Process results and create final analysis
        logger.info("Creating final aggregated analysis")
        final_prompt = aggregate_results(results)
        
        # Optionally validate the prompt
        # if input_text and output_text:
        #     logger.info("Starting prompt validation")
        #     validation_agent = Agent(
        #         name="ValidationAgent",
        #         instructions="Validate transformation prompt against input/output example",
        #         model="gpt-4o-mini"
        #     )
            
        #     validation_response = client.run(
        #         agent=validation_agent,
        #         messages=[{
        #             "role": "user",
        #             "content": f"""Validate if this prompt accurately captures the transformation:
                    
        #             PROMPT:
        #             {final_prompt}
                    
        #             INPUT EXAMPLE:
        #             {input_text}
                    
        #             EXPECTED OUTPUT:
        #             {output_text}
                    
        #             Provide feedback on prompt accuracy and completeness."""
        #         }]
        #     )
            
        #     validation_result = validation_response.messages[-1].get('content', 'No validation results available')
        #     logger.info("Validation completed")
        #     logger.info("-" * 80)
        #     logger.info("Validation Results:")
        #     logger.info(validation_result)
        #     logger.info("-" * 80)
        
        logger.info("Analysis pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main execution: {str(e)}", exc_info=True)
        raise

# if __name__ == "__main__":
#     logger.info("Starting prompt generator application")
#     asyncio.run(main())
#     logger.info("Application completed")
