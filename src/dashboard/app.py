"""Streamlit dashboard for earnings call alignment analysis."""

import os
from datetime import datetime

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Earnings Call Analyser",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")


def get_api_client() -> httpx.Client:
    """Get HTTP client for API calls."""
    return httpx.Client(base_url=API_URL, timeout=120.0)


def main() -> None:
    """Main dashboard application."""
    st.title("Earnings Call Q&A Alignment Analyser")

    st.markdown("""
    Analyse how well management answers analyst questions during earnings calls.
    **Low alignment (evasive answers) = negative signal for the stock.**

    Based on [Chiang et al. 2025](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608365/full) research.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Analyse Earnings Call",
            "Company Analysis",
            "Q&A Explorer",
            "Backtest Results",
        ],
    )

    if page == "Analyse Earnings Call":
        show_analyse_page()
    elif page == "Company Analysis":
        show_company_analysis()
    elif page == "Q&A Explorer":
        show_qa_explorer()
    elif page == "Backtest Results":
        show_backtest_results()


def show_analyse_page() -> None:
    """Show page to analyse new earnings calls."""
    st.header("Analyse Earnings Call")

    st.markdown("""
    Fetch and analyse recent earnings call transcripts. Enter a ticker symbol
    and optionally specify the quarter to analyse.
    """)

    # Show available data providers
    try:
        with get_api_client() as client:
            providers_resp = client.get("/api/providers")
            if providers_resp.status_code == 200:
                providers_data = providers_resp.json()
                available = providers_data.get("available_providers", [])
                primary = providers_data.get("primary_provider")
                if available:
                    st.info(f"Data sources: {', '.join(available)} (primary: {primary})")
                else:
                    st.warning("No API data sources configured. Use manual transcript input.")
    except Exception:
        pass  # Don't block UI if provider check fails

    # Popular tickers for quick selection
    st.subheader("Quick Select - Recent Q4 2025 Earnings")

    # Companies reporting in Jan/Feb 2025 for Q4 2024
    recent_earnings = {
        "Tech": ["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AMD", "INTC", "IBM"],
        "Financials": ["JPM", "BAC", "GS", "MS", "WFC", "C"],
        "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
        "Consumer": ["AMZN", "TSLA", "HD", "MCD", "NKE", "COST"],
    }

    cols = st.columns(4)
    for idx, (sector, tickers) in enumerate(recent_earnings.items()):
        with cols[idx]:
            st.markdown(f"**{sector}**")
            for ticker in tickers:
                if st.button(ticker, key=f"quick_{ticker}"):
                    st.session_state["selected_ticker"] = ticker

    st.divider()

    # Manual input
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        default_ticker = st.session_state.get("selected_ticker", "AAPL")
        ticker = st.text_input("Ticker Symbol", value=default_ticker).upper()

    with col2:
        year = st.selectbox("Year", [2025, 2024, 2023], index=0)

    with col3:
        quarter = st.selectbox("Quarter", [4, 3, 2, 1], index=0)

    # Input mode selection
    st.divider()
    input_mode = st.radio(
        "Transcript Source",
        ["Fetch from API", "Paste Transcript", "Upload File"],
        horizontal=True,
    )

    manual_transcript = None

    if input_mode == "Paste Transcript":
        st.markdown("**Paste your earnings call transcript below:**")
        manual_transcript = st.text_area(
            "Transcript Text",
            height=300,
            placeholder="Paste the full earnings call transcript here...\n\n"
            "You can copy this from:\n"
            "- Company investor relations website\n"
            "- Seeking Alpha (if you have access)\n"
            "- Financial news sources",
        )
        if manual_transcript and len(manual_transcript) < 500:
            st.warning("Transcript seems short. Ensure you've pasted the full Q&A section.")

    elif input_mode == "Upload File":
        uploaded_file = st.file_uploader(
            "Upload transcript file",
            type=["txt", "pdf"],
            help="Upload a .txt or .pdf file containing the earnings call transcript",
        )
        if uploaded_file is not None:
            if uploaded_file.type == "text/plain":
                manual_transcript = uploaded_file.read().decode("utf-8")
                st.success(f"Loaded {len(manual_transcript):,} characters from {uploaded_file.name}")
            elif uploaded_file.type == "application/pdf":
                try:
                    import PyPDF2
                    import io

                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                    text_parts = []
                    for page in pdf_reader.pages:
                        text_parts.append(page.extract_text())
                    manual_transcript = "\n".join(text_parts)
                    st.success(f"Extracted {len(manual_transcript):,} characters from {uploaded_file.name}")
                except ImportError:
                    st.error("PDF support requires PyPDF2. Install with: pip install PyPDF2")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

    analyse_btn = st.button("Analyse Transcript", type="primary")

    if analyse_btn and ticker:
        # Validate input
        if input_mode != "Fetch from API" and not manual_transcript:
            st.error("Please provide transcript text or upload a file.")
        else:
            with st.spinner(f"Fetching and analyzing {ticker} Q{quarter} {year} earnings call..."):
                try:
                    result = analyse_transcript(
                        ticker, year, quarter,
                        manual_transcript=manual_transcript if input_mode != "Fetch from API" else None
                    )
                    if result:
                        display_analysis_result(result)
                    else:
                        st.error("Analysis failed. Check the API logs for details.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def analyse_transcript(
    ticker: str, year: int, quarter: int, manual_transcript: str | None = None
) -> dict | None:
    """Call API to analyse transcript."""
    with get_api_client() as client:
        # Determine which endpoint to use
        if manual_transcript:
            response = client.post(
                "/api/analyse/manual",
                json={
                    "ticker": ticker,
                    "transcript": manual_transcript,
                    "fiscal_year": year,
                    "fiscal_quarter": quarter,
                },
            )
        else:
            response = client.post(
                "/api/analyse",
                json={
                    "ticker": ticker,
                    "fiscal_year": year,
                    "fiscal_quarter": quarter,
                    "fetch_from_api": True,
                },
            )

        if response.status_code != 200:
            st.error(f"API error: {response.text}")
            return None

        job_data = response.json()
        job_id = job_data.get("job_id")

        if not job_id:
            return job_data.get("result")

        # Poll for completion
        import time

        progress_bar = st.progress(0, text="Starting analysis...")

        for _ in range(60):  # Max 60 seconds
            time.sleep(1)

            status_response = client.get(f"/api/analyse/{job_id}")
            status_data = status_response.json()

            progress = status_data.get("progress", 0)
            message = status_data.get("message", "Processing...")
            progress_bar.progress(progress, text=message)

            if status_data.get("status") == "completed":
                progress_bar.progress(1.0, text="Complete!")
                return status_data.get("result")
            elif status_data.get("status") == "failed":
                st.error(f"Analysis failed: {status_data.get('message')}")
                return None

        st.error("Analysis timed out")
        return None


def display_analysis_result(result: dict) -> None:
    """Display analysis results."""
    data_source = result.get("data_source", "unknown")
    st.success(
        f"Analysis complete for {result['ticker']} Q{result['fiscal_quarter']} {result['fiscal_year']} "
        f"(Source: {data_source})"
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        score = result.get("overall_alignment_score", 0)
        color = "green" if score >= 0.7 else "orange" if score >= 0.4 else "red"
        st.metric("Overall Alignment", f"{score:.2f}")

    with col2:
        st.metric("Q&A Pairs Analysed", result.get("total_qa_pairs", 0))

    with col3:
        # Get dominant category
        categories = result.get("category_distribution", {})
        if categories:
            top_cat = max(categories, key=categories.get)
            st.metric("Top Question Category", top_cat.replace("_", " ").title())

    with col4:
        responders = result.get("responder_scores", {})
        if responders:
            top_resp = max(responders, key=responders.get)
            st.metric("Most Aligned Responder", top_resp.split()[0])

    st.divider()

    # Category distribution chart
    if result.get("category_distribution"):
        st.subheader("Question Category Distribution")

        cat_df = pd.DataFrame([
            {"Category": k.replace("_", " ").title(), "Count": v}
            for k, v in result["category_distribution"].items()
            if v > 0
        ])

        if not cat_df.empty:
            fig = px.pie(cat_df, values="Count", names="Category", hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Responder scores
    if result.get("responder_scores"):
        st.subheader("Alignment by Responder")

        resp_df = pd.DataFrame([
            {"Responder": k, "Alignment Score": v}
            for k, v in result["responder_scores"].items()
        ]).sort_values("Alignment Score", ascending=True)

        fig = px.bar(
            resp_df,
            x="Alignment Score",
            y="Responder",
            orientation="h",
            color="Alignment Score",
            color_continuous_scale=["red", "yellow", "green"],
            range_color=[0, 1],
        )
        fig.update_layout(height=max(200, len(resp_df) * 40))
        st.plotly_chart(fig, use_container_width=True)

    # Q&A pairs table
    if result.get("qa_pairs"):
        st.subheader("Q&A Pairs Analysis")

        qa_df = pd.DataFrame(result["qa_pairs"])
        qa_df["question_preview"] = qa_df["question_text"].str[:100] + "..."
        qa_df["answer_preview"] = qa_df["answer_text"].str[:100] + "..."

        # Score distribution
        fig = px.histogram(
            qa_df,
            x="alignment_score",
            nbins=20,
            title="Alignment Score Distribution",
            labels={"alignment_score": "Alignment Score", "count": "Count"},
        )
        fig.add_vline(x=0.7, line_dash="dash", line_color="green", annotation_text="Aligned")
        fig.add_vline(x=0.3, line_dash="dash", line_color="red", annotation_text="Evasive")
        st.plotly_chart(fig, use_container_width=True)

        # Detailed Q&A view
        st.subheader("Individual Q&A Pairs")

        sort_by = st.selectbox(
            "Sort by",
            ["alignment_score", "sequence_number"],
            format_func=lambda x: "Alignment Score" if x == "alignment_score" else "Order in Call"
        )
        ascending = st.checkbox("Ascending", value=sort_by == "sequence_number")

        sorted_qa = qa_df.sort_values(sort_by, ascending=ascending)

        for _, qa in sorted_qa.iterrows():
            score = qa["alignment_score"]
            label = qa["alignment_label"]

            color = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"

            with st.expander(
                f"{color} {qa['analyst_name']} → {qa['responder_name']} | "
                f"Score: {score:.2f} | {qa['question_category'].replace('_', ' ').title()}"
            ):
                st.markdown(f"**Question:** {qa['question_text']}")
                st.markdown(f"**Answer:** {qa['answer_text']}")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Alignment Score", f"{score:.2f}")
                with col2:
                    st.metric("Label", label.replace("_", " ").title())
                with col3:
                    st.metric("Cosine Similarity", f"{qa['cosine_similarity']:.3f}")


def show_company_analysis() -> None:
    """Show company-specific analysis."""
    st.header("Company Analysis")

    st.info("""
    **To see live data:** Use the "Analyse Earnings Call" page to analyse recent transcripts.
    Results will be stored and shown here for comparison.

    The alignment timeline below shows sample data. Analyse multiple quarters for a company
    to build up historical data.
    """)

    # Company selector
    col1, col2 = st.columns([1, 3])

    with col1:
        ticker = st.text_input("Enter Ticker", value="AAPL").upper()

    with col2:
        st.markdown(f"### {ticker}")

    show_alignment_timeline(ticker)


def show_alignment_timeline(ticker: str) -> None:
    """Show alignment timeline for a company."""
    st.subheader(f"Alignment Timeline - {ticker}")

    # Sample data for demonstration
    # In production, this would come from the database via API
    quarters = ["Q1'25", "Q2'25", "Q3'25", "Q4'25"]
    alignment_scores = [0.65, 0.62, 0.68, 0.71]
    stock_returns = [0.05, -0.02, 0.08, 0.12]

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Alignment score line
    fig.add_trace(
        go.Scatter(
            x=quarters,
            y=alignment_scores,
            name="Alignment Score",
            line=dict(color="#2E86AB", width=3),
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    # Stock return bars
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in stock_returns]
    fig.add_trace(
        go.Bar(
            x=quarters,
            y=[r * 100 for r in stock_returns],
            name="30-Day Return (%)",
            marker_color=colors,
            opacity=0.6,
        ),
        secondary_y=True,
    )

    # Add threshold lines
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Aligned")
    fig.add_hline(y=0.3, line_dash="dash", line_color="red", annotation_text="Evasive")

    fig.update_layout(
        title=f"{ticker} - Alignment Score vs Stock Performance",
        height=500,
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Alignment Score", secondary_y=False, range=[0, 1])
    fig.update_yaxes(title_text="30-Day Return (%)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: This shows sample data. Analyse multiple quarters to build historical data.")


def show_qa_explorer() -> None:
    """Show Q&A pair explorer."""
    st.header("Q&A Pair Explorer")

    st.info("""
    After analyzing earnings calls on the "Analyse Earnings Call" page,
    you can explore individual Q&A pairs here.
    """)

    # Check if we have analysis results in session state
    if "last_analysis" not in st.session_state:
        st.warning("No analysis results available. Go to 'Analyse Earnings Call' to analyse a transcript first.")

        # Show sample for demonstration
        st.subheader("Sample Q&A Pairs (for demonstration)")

        sample_qa = [
            {
                "analyst": "John Smith - Goldman Sachs",
                "question": "Can you provide more color on the margin expansion in Services?",
                "category": "Margins",
                "responder": "CFO",
                "answer": "Yes, the Services margin improvement is driven by scale benefits and operational efficiencies.",
                "score": 0.82,
                "label": "aligned",
            },
            {
                "analyst": "Sarah Chen - Morgan Stanley",
                "question": "How are you thinking about the competitive landscape in AI?",
                "category": "Competition",
                "responder": "CEO",
                "answer": "We focus on our own path and making the best products for our customers.",
                "score": 0.41,
                "label": "partially_aligned",
            },
        ]

        for qa in sample_qa:
            color = "🟢" if qa["score"] >= 0.7 else "🟡" if qa["score"] >= 0.4 else "🔴"
            with st.expander(f"{color} {qa['analyst']} - {qa['category']} (Score: {qa['score']:.2f})"):
                st.markdown(f"**Question:** {qa['question']}")
                st.markdown(f"**Responder:** {qa['responder']}")
                st.markdown(f"**Answer:** {qa['answer']}")
                st.progress(qa["score"], text=f"Alignment: {qa['score']:.2f}")


def show_backtest_results() -> None:
    """Show backtest results."""
    import numpy as np

    st.header("Backtest Results")

    st.markdown("""
    **Strategy:** Long companies with high alignment (>0.7), short companies with low alignment (<0.3)

    This shows hypothetical backtest results based on the research paper findings.
    To run actual backtests, you need to analyse multiple companies over multiple quarters.
    """)

    # Configuration
    with st.expander("Backtest Configuration"):
        col1, col2, col3 = st.columns(3)

        with col1:
            high_threshold = st.slider("High Alignment Threshold", 0.5, 0.9, 0.7)

        with col2:
            low_threshold = st.slider("Low Alignment Threshold", 0.1, 0.5, 0.3)

        with col3:
            holding_period = st.selectbox("Holding Period", [30, 60, 90], index=0)

    # Results
    st.subheader("Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Return", "+47.3%", delta="+12.1% vs S&P 500")

    with col2:
        st.metric("Sharpe Ratio", "1.24")

    with col3:
        st.metric("Win Rate", "58.2%")

    with col4:
        st.metric("Max Drawdown", "-18.4%")

    # Statistical significance
    st.subheader("Statistical Significance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Correlation (Score vs Return)", "0.34")

    with col2:
        st.metric("P-Value", "0.003", help="< 0.05 indicates statistical significance")

    with col3:
        st.metric("R-Squared", "0.116")

    st.success("The alignment signal shows statistically significant predictive power (p < 0.05)")

    # Equity curve
    st.subheader("Equity Curve")

    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    strategy_returns = [1.0, 1.02, 1.01, 1.05, 1.08, 1.06, 1.12, 1.15, 1.13, 1.18, 1.22, 1.20,
                        1.25, 1.28, 1.26, 1.32, 1.35, 1.33, 1.38, 1.42, 1.40, 1.45, 1.47, 1.47]
    benchmark_returns = [1.0, 1.01, 0.99, 1.02, 1.04, 1.03, 1.06, 1.08, 1.05, 1.09, 1.12, 1.10,
                        1.13, 1.15, 1.12, 1.16, 1.18, 1.15, 1.19, 1.22, 1.20, 1.24, 1.27, 1.25]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=dates, y=strategy_returns, name="Alignment Strategy", line=dict(color="#2E86AB", width=2))
    )
    fig.add_trace(
        go.Scatter(x=dates, y=benchmark_returns, name="S&P 500", line=dict(color="#999999", width=2, dash="dash"))
    )

    fig.update_layout(title="Strategy vs Benchmark", xaxis_title="Date", yaxis_title="Cumulative Return", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.subheader("Alignment Score vs 30-Day Return")

    np.random.seed(42)
    n_points = 100
    scores = np.random.uniform(0.2, 0.9, n_points)
    returns = 0.3 * (scores - 0.5) + np.random.normal(0, 0.08, n_points)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=scores,
            y=returns * 100,
            mode="markers",
            marker=dict(color=scores, colorscale=["#F44336", "#FFC107", "#4CAF50"], size=8, opacity=0.7),
            name="Q&A Pairs",
        )
    )

    # Add trend line
    z = np.polyfit(scores, returns * 100, 1)
    p = np.poly1d(z)
    fig.add_trace(
        go.Scatter(x=[0.2, 0.9], y=[p(0.2), p(0.9)], mode="lines", line=dict(color="black", dash="dash"), name="Trend")
    )

    fig.update_layout(title="Alignment Score vs Forward Return", xaxis_title="Alignment Score",
                      yaxis_title="30-Day Forward Return (%)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Note: This shows simulated data based on research findings. Run actual analysis to generate real results.")


if __name__ == "__main__":
    main()
