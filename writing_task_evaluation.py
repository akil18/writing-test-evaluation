import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Get the GROQ API key from environment variables
groq_api_key = os.getenv("WTE_GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("WTE_GROQ_API_KEY is not set. Please check your .env file.")
model = os.getenv("MODEL")
if not model:
    raise ValueError("MODEL is not set. Please check your .env file.")


# Use the API key in the ChatGroq model
llm = ChatGroq(
    temperature=0,
    groq_api_key=groq_api_key,
    model=model,
)

# response = llm.invoke("The second person to land on the moon was...")
# print(response.content)

def get_criteria():
    # Load evaluation criteria from band_assignments.json file
    with open('band_assignments.json', 'r') as file:
        evaluation_criteria = json.load(file)

    # Prepare the evaluation criteria as a string for the prompt
    criteria_string = ""
    for band in evaluation_criteria['bands']:
        criteria_string += f"{band['band']}:\n"
        for key, value in band['criteria'].items():
            criteria_string += f"  {key}: {value}\n"
    
    return criteria_string

def evaluate(writing_sample, writing_question, criteria_string):

    # Define the prompt template using the formatted criteria string
    evaluation_prompt = PromptTemplate.from_template(
        f"""
        ### WRITING SAMPLE:
        {writing_sample}

        ### QUESTION:
        {writing_question}

        ### IELTS WRITING TEST INFORMATION:
        The IELTS Writing test evaluates your ability to express yourself in written English and is designed to assess your writing skills in an academic or general context, depending on whether you are taking the Academic or General Training version of the IELTS test. The {writing_sample} needs to address the {writing_question} and failing to do so will result in a decrease in score. If the {writing_sample} does not meet the requirements for word count, the score will be affected.

        **Test Format**:
        - The IELTS Writing test consists of two tasks:
        - **Task 1**: 
            - Academic: Describe visual information, such as graphs, charts, or diagrams.
            - General Training: Write a letter based on a given situation.
        - **Task 2**: Write an essay in response to a question or statement.

        **Word Count**:
        - Task 1: At least 150 words.
        - Task 2: At least 250 words.

        **Skills Assessed**:
        - Task 1: Present and describe visual information logically and coherently (Academic) or convey information in a clear and appropriate tone (General Training).
        - Task 2: Provide well-structured arguments, opinions, and ideas in written form.

        ### EVALUATION CRITERIA:
        {criteria_string}

        ### INSTRUCTION:
        Evaluate the writing sample based on the provided criteria and score it on a scale from 1 to 9. Use the following steps:
        1. **Task Type Context**: Determine whether the task is Task 1 or Task 2 based on the {writing_question} and evaluate accordingly.
        2. **Task Response**: Assess how well the response addresses the question, its relevance, and development of ideas.
        3. **Coherence and Cohesion**: Evaluate the logical flow, organization of ideas, and use of cohesive devices.
        4. **Lexical Resource**: Examine the range, accuracy, and appropriateness of vocabulary used.
        5. **Grammatical Range and Accuracy**: Analyze the sentence structures, grammar, and punctuation for range and accuracy.

        ### Provide a JSON response(NO PREAMBLE):
        - `score`: The overall band score (1-9).
        - `reasoning`: A detailed explanation for the score, referencing specific points from the writing sample and evaluation criteria.
        - `task_type`: State what task type it was.
        """
    )

    chain_evaluation = evaluation_prompt | llm
    res = chain_evaluation.invoke({
        "writing_sample": writing_sample, 
        "writing_question": writing_question
    })
    return res.content


if __name__ == "__main__":
    # Example usage of the prompt with a writing sample and question
    # writing_sample = "Social networking sites, for instance Facebook, are thought by some to have had a detrimental effect on individual people as well as society and local communities. However, while I believe that such sites are mainly beneficial to the individual, I agree that they have had a damaging effect on local communities. With regards to individuals, the impact that online social media has had on each individual person has clear advantages. Firstly, people from different countries are brought together through such sites as Facebook whereas before the development of technology and social networking sites, people rarely had the chance to meet or communicate with anyone outside of their immediate circle or community. Secondly, Facebook also has social groups which offer individuals a chance to meet and participate in discussions with people who share common interests. On the other hand, the effect that Facebook and other social networking sites have had on societies and local communities can only be seen as negative. Rather than individual people taking part in their local community, they are instead choosing to take more interest in people online. Consequently, the people within local communities are no longer forming close or supportive relationships. Furthermore, society as a whole is becoming increasingly disjointed and fragmented as people spend more time online with people they have never met face to face and who they are unlikely to ever meet in the future. To conclude, although social networking sites have brought individuals closer together, they have not had the same effect on society or local communities. Local communities should do more to try and involve local people in local activities  in order to promote the future of community life."
    writing_sample = "The pie charts show the amount of revenue and expenditures in 2016 for a children’s charity in the USA. Overall, it can be seen that donated food accounted for the majority of the income, while program services accounted for the most expenditure. Total revenue sources just exceeded outgoings. In detail, donated food provided most of the revenue for the charity, at 86%. Similarly, with regard to expenditures, one category, program services, accounted for nearly all of the outgoings, at 95.8%. The other categories were much smaller. Community contributions, which were the second largest revenue source, brought in 10.4% of overall income, and this was followed by program revenue, at 2.2%. Investment income, government grants, and other income were very small sources of revenue, accounting for only 0.8% combined. There were only two other expenditure items, fundraising and management and general, accounting for 2.6% and 1.6% respectively. The total amount of income was $53,561,580, which was just enough to cover the expenditures of $53,224,896."
    # writing_question = "Many people believe that social networking sites (such as Facebook) have had a huge negative impact on both individuals and society. To what extent do you agree?"
    writing_question = "The pie chart shows the amount of money that a children's charity located in the USA spent and received in one year, 2016. Summarise the information by selecting and reporting the main features and make comparisons where relevant. Write at least 150 words."
    criteria_string = get_criteria()
    # print("criteria string: ", criteria_string)
    evaluation = evaluate(writing_sample, writing_question, criteria_string)
    print(evaluation)