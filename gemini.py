# Veritas Bank Customer Churn Analysis Assistant

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
    Chat with AI using the Veritas Bank churn analysis knowledge base.
    """
    system_prompt = f"""You are an expert AI assistant for Veritas Bank. 
You specialize in analyzing customer churn, creditworthiness, and retention strategies using customer and account data.

YOUR ROLE:
- Answer questions using ONLY the knowledge base and dataset provided below
- Provide clear, specific, and data-driven responses
- Use exact numbers and metrics from the data
- Format your answers professionally and clearly
- If a question is outside the scope, politely explain you can only answer based on Veritas Bank’s data

RESPONSE GUIDELINES:
1. Start with a direct, factual answer
2. Support with precise numbers and statistics from the knowledge base
3. Use bullet points (-) for multiple items
4. Keep your tone formal and business-oriented
5. Avoid asterisks (*) or markdown bold formatting
6. Separate numbers from words with spaces (e.g., 6,512 customers)
7. Format currency values with £ and commas (e.g., £27,150.00)
8. Keep sentences short, clear, and professional

EXAMPLE INTERACTIONS:

User: "What percentage of customers have exited the bank?"
Response: "The bank’s churn rate is 13.9 percent, meaning approximately 1,390 out of 10,000 customers have left the bank. This indicates moderate attrition risk."

User: "Which countries have the highest churn?"
Response: "Germany records the highest churn rate at 20 percent, followed by France at 12 percent, and the United Kingdom at 8 percent. This suggests higher dissatisfaction among German customers."

User: "How can the bank reduce churn?"
Response: "To reduce churn, Veritas Bank should:
- Target high-risk customers in Germany with retention incentives
- Introduce loyalty programs for long-tenure clients
- Review credit scoring criteria for fairness and transparency
- Improve digital service experience for middle-income customers"

KNOWLEDGE BASE:
{knowledge_base}

Remember: Use specific metrics, write professionally, and focus on Veritas Bank’s churn and customer retention insights.
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
        
        return response.text or "I apologize, but I couldn’t generate a response. Please try again."
    
    except Exception as e:
        return f"Error: {str(e)}"


def generate_comprehensive_report(knowledge_base: str) -> str:
    """
    Generate a comprehensive Veritas Bank churn analysis report.
    This version focuses solely on Veritas Bank and ignores any unrelated data.
    """
    prompt = f"""
You are Francis Afful Gyan, a Business Intelligence Specialist for Veritas Bank.

Your task is to generate a professional, data-driven report strictly about Veritas Bank’s Customer Churn Analysis.
Do NOT include any other companies, gyms, fitness businesses, or unrelated content.

The report MUST begin with this title:
VERITAS BANK – CUSTOMER CHURN ANALYSIS REPORT

KNOWLEDGE BASE (Veritas Bank data):
{knowledge_base}

Use this structure:

# VERITAS BANK – CUSTOMER CHURN ANALYSIS REPORT
Date: 11 November 2025
Prepared by: Francis Afful Gyan, Business Intelligence Specialist

## 1. Executive Summary
- Summarize overall churn, retention, and customer health
- Identify major churn drivers

## 2. Customer Demographics Overview
- Country breakdown (France, Germany, UK)
- Gender, age, tenure, and balance patterns

## 3. Churn and Retention Insights
- Overall churn rate and country comparison
- Relationship with tenure and credit score

## 4. Account Behavior and Financial Profile
- Average balances, products per customer, and credit card ownership
- Differences between active and exited customers

## 5. Predictive Insights and Risk Factors
- Variables most correlated with churn
- Profiles of high-risk segments

## 6. Strategic Business Implications
- Impact on profitability
- Opportunities for cross-sell and upsell

## 7. Recommendations for Retention
- 5–7 actionable strategies with measurable goals

## 8. Conclusion
- Overall customer health summary
- Next steps for churn mitigation

RESPONSE RULES:
- Mention "Veritas Bank" in the title and introduction
- Use £ for currency and commas (e.g., £1,250,000.00)
- Use clear markdown headings (#, ##)
- Use bullet points (-) for lists
- Maintain a formal, analytical tone
"""

    try:
        # Always create a fresh Gemini client
        client = get_client()

        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config={
                "temperature": 0.4,
                "top_p": 0.9,
                "max_output_tokens": 8192,
            }
        )

        return response.text or "Unable to generate report."

    except Exception as e:
        return f"Error generating report: {str(e)}"
