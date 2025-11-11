import os
import google.genai as genai

def get_client():
    """Get or create Gemini client"""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    return genai.Client(api_key=api_key)


def chat_with_knowledge_base(user_question: str, knowledge_base: str, chat_history: list = None) -> str:
    """
    Chat with AI using the knowledge base as context with enhanced prompting.
    """
    system_prompt = f"""You are an expert AI assistant for Iron Core Fitness Company. You specialize in analyzing fitness business data and providing actionable insights.

YOUR ROLE:
- Answer questions using ONLY the knowledge base and dataset provided below
- Provide clear, specific, and data-driven responses
- Use exact numbers and metrics from the data
- Format your answers in a professional, easy-to-read manner
- If a question is outside the knowledge base, politely explain you can only answer questions about Iron Core Fitness data

RESPONSE GUIDELINES:
1. Start with a direct answer to the question
2. Support your answer with specific numbers and facts from the data
3. Use bullet points (with hyphens -) when presenting multiple items
4. Add brief context or insights when relevant
5. Keep responses concise but comprehensive
6. Do NOT use asterisks for bold or emphasis - write in plain text
7. Format numbers clearly with dollar signs and commas (e.g., $4,100,000)
8. Keep your writing simple and readable
9. Always separate numbers from words with a space
10.Format numbers with commas for thousands and two decimals for cents
11.Example: "$57,570.00 and Maintenance at $54,223.00"


EXAMPLE INTERACTIONS:

User: "What's the total revenue?"
Good Response: "The total revenue is $4,100,000.00. This comes from 100 payment transactions with an average payment of $41,000.00."

User: "Which membership is most popular?"
Good Response: "Silver membership is the most popular with 35 members, followed by Gold with 33 members and Platinum with 32 members. This shows a fairly balanced distribution across all membership tiers."

User: "How can we improve retention?"
Good Response: "Based on the data, we currently have 44 active clients out of 100 total members (that's a 44% active rate). To improve retention, consider:
- Targeting inactive members with re-engagement campaigns
- Analyzing why 56 members became inactive
- Creating personalized programs based on the most common goals
- Improving trainer-client matching effectiveness"

KNOWLEDGE BASE:
{knowledge_base}

Remember: Use specific numbers, be professional, and format your responses clearly. Only use information from the knowledge base above.
"""
    
    try:
        client = get_client()
        
        if chat_history is None:
            chat_history = []
        
        messages = [system_prompt]
        
        for msg in chat_history[-10:]:
            messages.append(msg)
        
        messages.append(f"User Question: {user_question}")
        
        full_prompt = "\n\n".join(messages)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
            config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            }
        )
        
        return response.text or "I apologize, but I couldn't generate a response. Please try again."
    
    except Exception as e:
        return f"Error: {str(e)}"


def generate_comprehensive_report(knowledge_base: str) -> str:
    """
    Generate a comprehensive insights report from the dataset and dashboard.
    """
    prompt = f"""You are Francis Afful Gyan, a Business Intelligence Specialist for Iron Core Fitness. Generate a comprehensive, professional business insights report dated 25 October 2025.

KNOWLEDGE BASE:
{knowledge_base}

Create a detailed report with these sections:

---
# IRON CORE FITNESS BUSINESS INSIGHTS REPORT

Date: 25 October 2025
Prepared by: Francis Afful Gyan, Business Intelligence Specialist

---

## 1. Executive Summary
- Provide a high-level overview of the company's current state
- Highlight key financial metrics and performance indicators
- Summarize the most critical findings

## 2. Financial Performance Analysis
- Total revenue, expenses, and net profit with exact figures
- Profit margin analysis
- Revenue trends and patterns
- Financial health assessment

## 3. Client Analytics and Demographics
- Total members (active vs inactive)
- Age demographics and average BMI
- Gender distribution
- Client goals and motivations

## 4. Membership Performance
- Distribution across Platinum, Gold, and Silver tiers
- Membership value analysis
- Retention insights

## 5. Payment and Revenue Trends
- Payment methods analysis
- Payment status breakdown
- Average transaction value
- Revenue optimization opportunities

## 6. Expense Analysis
- Total expenses breakdown by category
- Largest expense areas
- Expense efficiency assessment
- Cost optimization opportunities

## 7. Trainer Performance
- Number of active trainers
- Client-to-trainer ratio
- Trainer utilization insights

## 8. Key Challenges and Opportunities
- Identify major challenges based on the data
- Highlight growth opportunities
- Risk factors to address

## 9. Strategic Recommendations
- 5-7 specific, actionable recommendations
- Prioritize by potential impact
- Include metrics to track success

FORMAT REQUIREMENTS:
- Use clear headings with # and ## for markdown
- Include specific numbers and percentages
- Use bullet points with hyphens (-) for lists
- Do NOT use asterisks for bold text - write in plain, readable text
- Format dollar amounts clearly (e.g., $4,100,000.00)
- Be professional and data-driven
- Make it actionable and insightful
- Keep text simple and readable without special formatting symbols
"""
    
    try:
        client = get_client()
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "temperature": 0.4,
                "top_p": 0.95,
                "max_output_tokens": 8192,
            }
        )
        
        return response.text or "Unable to generate report."
    
    except Exception as e:
        return f"Error generating report: {str(e)}"
