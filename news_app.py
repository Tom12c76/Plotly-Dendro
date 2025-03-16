import streamlit as st
import pandas as pd
import os
import eikon as ek
from openai import OpenAI
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Dow Jones 30 Portfolio Commentary",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Dow Jones 30 Portfolio Commentary Generator")
st.markdown("This app generates portfolio commentary for the Dow Jones 30 Industrial Average based on recent news")

# Dow Jones 30 Companies
dow_jones_tickers = {
    "AAPL.O": "Apple Inc",
    "AMGN.O": "Amgen Inc",
    "AXP.N": "American Express Co",
    "BA.N": "Boeing Co",
    "CAT.N": "Caterpillar Inc",
    "CRM.N": "Salesforce Inc",
    "CSCO.O": "Cisco Systems Inc",
    "CVX.N": "Chevron Corp",
    "DIS.N": "Walt Disney Co",
    "DOW.N": "Dow Inc",
    "GS.N": "Goldman Sachs Group Inc",
    "HD.N": "Home Depot Inc",
    "HON.N": "Honeywell International Inc",
    "IBM.N": "International Business Machines Corp",
    "INTC.O": "Intel Corp",
    "JNJ.N": "Johnson & Johnson",
    "JPM.N": "JPMorgan Chase & Co",
    "KO.N": "Coca-Cola Co",
    "MCD.N": "McDonald's Corp",
    "MMM.N": "3M Co",
    "MRK.N": "Merck & Co Inc",
    "MSFT.O": "Microsoft Corp",
    "NKE.N": "Nike Inc",
    "PG.N": "Procter & Gamble Co",
    "TRV.N": "Travelers Companies Inc",
    "UNH.N": "UnitedHealth Group Inc",
    "V.N": "Visa Inc",
    "VZ.N": "Verizon Communications Inc",
    "WBA.O": "Walgreens Boots Alliance Inc",
    "WMT.N": "Walmart Inc"
}

# Function to display tickers
def display_tickers():
    st.subheader("Dow Jones 30 Components")
    ticker_df = pd.DataFrame(list(dow_jones_tickers.items()), columns=['Ticker', 'Company'])
    st.dataframe(ticker_df)

# Function to fetch news from Eikon
@st.cache_data(ttl=3600)
def fetch_news_for_ticker(ticker, company_name, days=7, max_items=5):
    """Fetch news for a specific ticker from Eikon"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for Eikon API
        start_date_str = start_date.strftime('%Y-%m-%dT%H:%M:%S')
        end_date_str = end_date.strftime('%Y-%m-%dT%H:%M:%S')
        
        # Query Eikon for news - use unpacking with * to handle variable number of return values
        api_result = ek.get_news_headlines(
            query=f'R:{ticker} and Language:en',
            count=max_items,
            date_from=start_date_str,
            date_to=end_date_str
        )
        
        # The first element in the return value is always the news_headlines DataFrame
        news_headlines = api_result[0] if isinstance(api_result, tuple) else api_result
        
        # Add company name to results
        if not news_headlines.empty:
            news_headlines['company'] = company_name
        
        return news_headlines
    except Exception as e:
        st.error(f"Error fetching news for {ticker} ({company_name}): {e}")
        return pd.DataFrame()

# Function to fetch news stories (content)
@st.cache_data(ttl=3600)
def fetch_news_content(story_id):
    """Fetch the full content of a news story"""
    try:
        news_story = ek.get_news_story(story_id)
        return news_story
    except Exception as e:
        st.error(f"Error fetching news content for story {story_id}: {e}")
        return None

# Function to extract relevant information from news
def process_news_data(news_df):
    """Process the news dataframe to extract relevant information"""
    processed_data = []
    
    for _, row in news_df.iterrows():
        try:
            # Fetch full news content
            story_content = fetch_news_content(row['storyId'])
            
            # Extract important information
            news_item = {
                'company': row['company'],
                'ticker': row.get('sourceCode', ''),
                'headline': row['text'],
                'date': row['versionCreated'],
                'content': story_content,
                'sentiment': 'neutral'  # Default sentiment, could be analyzed
            }
            
            processed_data.append(news_item)
            
        except Exception as e:
            st.warning(f"Error processing news item: {e}")
    
    return processed_data

# Initialize OpenAI client
client = None

# Function to generate commentary with OpenAI
def generate_company_commentary(company_name, news_data):
    """Generate commentary for a single company using GPT"""
    if not news_data:
        return f"No recent significant news for {company_name}."
    
    # Create a prompt for OpenAI
    news_text = "\n\n".join([
        f"Headline: {item['headline']}\n" + 
        f"Date: {item['date']}\n" +
        f"Content: {item.get('content', 'No content available')[:500]}..."  # Limit content length
        for item in news_data
    ])
    
    prompt = f"""
    As an expert financial analyst, provide a concise analysis (2-3 paragraphs) of {company_name} based on the following recent news:
    
    {news_text}
    
    Focus on:
    1. Key business developments
    2. Market perception and sentiment
    3. Potential impact on stock performance
    
    Keep your analysis factual, balanced, and informative.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Or "gpt-3.5-turbo" depending on your subscription
            messages=[
                {"role": "system", "content": "You are an expert financial analyst specializing in equity markets and company analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating commentary for {company_name}: {e}")
        return f"Unable to generate commentary for {company_name} due to an error."

# Function to generate portfolio-level summary
def generate_portfolio_summary(company_commentaries, portfolio_name="Dow Jones 30"):
    """Generate a portfolio-level summary based on individual company commentaries"""
    
    # Create a combined text of all company commentaries
    combined_text = "\n\n".join(f"{company}: {commentary}" for company, commentary in company_commentaries.items())
    
    prompt = f"""
    As an expert portfolio manager, write a comprehensive market commentary (3-4 paragraphs) on the {portfolio_name} portfolio based on the following individual company analyses:
    
    {combined_text}
    
    In your commentary:
    1. Identify major themes and trends across the portfolio
    2. Highlight sectors showing strength or weakness
    3. Discuss overall market sentiment and outlook
    4. Note any significant risks or opportunities
    
    The commentary should be balanced, insightful, and suitable for a quarterly investor report.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Or "gpt-3.5-turbo" depending on your subscription
            messages=[
                {"role": "system", "content": "You are an expert portfolio manager with deep knowledge of equity markets and macroeconomic trends."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating portfolio summary: {e}")
        return "Unable to generate portfolio summary due to an error."

# Sidebar for API keys
with st.sidebar:
    st.subheader("API Configuration")
    
    eikon_api_key = st.text_input("Eikon API Key", type="password")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    
    if eikon_api_key:
        try:
            ek.set_app_key(eikon_api_key)
            st.success("Eikon API Key set successfully!")
        except Exception as e:
            st.error(f"Error setting Eikon API key: {e}")
    
    if openai_api_key:
        try:
            # Initialize OpenAI client with the API key
            client = OpenAI(api_key=openai_api_key)
            st.success("OpenAI API Key set successfully!")
        except Exception as e:
            st.error(f"Error setting OpenAI API key: {e}")
    
    # News fetching parameters
    st.subheader("News Parameters")
    days_to_look_back = st.slider("Days to look back", 1, 30, 7)
    news_per_company = st.slider("Max news items per company", 1, 10, 5)
    selected_companies = st.multiselect(
        "Select specific companies (leave empty for all)",
        list(dow_jones_tickers.values()),
        []
    )

# Display tickers on the main page
display_tickers()

# Main functionality
st.subheader("News and Commentary")

if not eikon_api_key or not openai_api_key:
    st.info("Please enter your API keys in the sidebar to fetch news and generate commentary.")
else:
    # Generate button
    if st.button("Fetch News and Generate Commentary"):
        with st.spinner("Fetching news from Eikon..."):
            # Determine which companies to process
            companies_to_process = {}
            if selected_companies:
                # Create a reverse mapping from company name to ticker
                reverse_mapping = {v: k for k, v in dow_jones_tickers.items()}
                companies_to_process = {reverse_mapping[company]: company for company in selected_companies}
            else:
                companies_to_process = dow_jones_tickers
            
            # Initialize container for all news
            all_news_df = pd.DataFrame()
            
            # Process each company
            for ticker, company in companies_to_process.items():
                st.text(f"Fetching news for {company}...")
                company_news = fetch_news_for_ticker(ticker, company, days=days_to_look_back, max_items=news_per_company)
                
                if not company_news.empty:
                    all_news_df = pd.concat([all_news_df, company_news])
        
        # Process news data
        if all_news_df.empty:
            st.warning("No news found for the selected companies and time period.")
        else:
            with st.spinner("Processing news and extracting information..."):
                processed_news = process_news_data(all_news_df)
                
                # Group news by company
                news_by_company = {}
                for item in processed_news:
                    company = item['company']
                    if company not in news_by_company:
                        news_by_company[company] = []
                    news_by_company[company].append(item)
                
                # Generate commentary for each company
                company_commentaries = {}
                
                for company, news in news_by_company.items():
                    with st.spinner(f"Generating commentary for {company}..."):
                        commentary = generate_company_commentary(company, news)
                        company_commentaries[company] = commentary
                
                # Display individual company commentaries
                st.subheader("Company Commentaries")
                if company_commentaries:
                    tabs = st.tabs(list(company_commentaries.keys()))
                    for i, (company, commentary) in enumerate(company_commentaries.items()):
                        with tabs[i]:
                            st.write(commentary)
                            
                            # Display raw news for reference
                            with st.expander("View Raw News"):
                                for news_item in news_by_company.get(company, []):
                                    st.markdown(f"**{news_item['headline']}**")
                                    st.text(f"Date: {news_item['date'].strftime('%Y-%m-%d')}")
                                    st.text("---")
                else:
                    st.warning("No commentaries were generated. Try adjusting your search parameters.")
                
                # Generate and display portfolio summary
                if company_commentaries:
                    with st.spinner("Generating portfolio summary..."):
                        st.subheader("Portfolio Commentary")
                        portfolio_summary = generate_portfolio_summary(company_commentaries)
                        st.markdown(portfolio_summary)
                        
                        # Add download button for the summary
                        st.download_button(
                            label="Download Portfolio Commentary",
                            data=portfolio_summary,
                            file_name=f"dow_jones_commentary_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )