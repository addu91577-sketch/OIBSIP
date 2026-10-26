# 🎬 Video Presentation Guide & Narration Script
## Retail Sales Exploratory Data Analysis (EDA) — Oasis Infobyte Task Level 1

This guide provides everything you need to record an executive-level video presentation for your Retail Sales EDA project. It includes **what to show on screen**, **what to speak word-for-word**, **timestamps**, and **ready-to-use AI prompts** for slide creation, voiceover, and video tools.

---

## 🚀 Quick Start: 3 Ways to Record Your Video

| Method | Best For | How To Do It |
| :--- | :--- | :--- |
| **Option A: Interactive Slide Deck (Recommended)** | High visual appeal, executive feel, zero slide prep | 1. Double-click [`presentation/index.html`](presentation/index.html) to open in your browser.<br>2. Press **`F`** or **`F11`** for Fullscreen.<br>3. Press **`N`** if you want to toggle speaker notes.<br>4. Open **OBS Studio**, **Loom**, or **Windows Game Bar (`Win + G`)** and record your screen while advancing with the arrow keys. |
| **Option B: Jupyter Notebook Walkthrough** | Technical audiences, code reviewers, academic grading | 1. Open `Notebooks/01_retail_sales_eda.ipynb` in Jupyter Notebook / VS Code.<br>2. Scroll through markdown summaries and executed charts as you speak. |
| **Option C: AI Video Generator** | Automated presentation video (no microphone required) | Use our [AI Video Generation Prompts](#-ai-prompts-for-presentation-tools) in **Gamma.app**, **ElevenLabs**, or **HeyGen**. |

---

## ⏱️ Recommended Video Duration

* **Standard Executive Presentation:** **4 to 6 minutes** (Covering all 12 slides at ~25–30 seconds each).
* **Elevator Pitch / Lightning Talk:** **2 to 3 minutes** (Covering Slides 1, 3, 6, 10, and 12).
* **Deep-Dive Technical Presentation:** **8 to 10 minutes** (Reviewing code cells and statistical tests).

---

## 🎥 Slide-by-Slide Visual Storyboard & Script

---

### Slide 1: Title & Executive Briefing
* **Estimated Time:** 0:00 – 0:30 (30 seconds)
* **What to Show on Screen:**
  - Slide 1 of [`presentation/index.html`](presentation/index.html) displaying the title *"Retail Sales Exploratory Data Analysis"*, author name, and the 4 metric cards ($456k Gross Turnover, 1,000 Transactions, 100% Data Integrity, Gini 0.6091).
* **What to Speak (Script):**
  > *"Hello everyone! Welcome to my presentation on this end-to-end Exploratory Data Analysis of retail sales performance.*
  > 
  > *In this project, our primary goal was to look beyond raw transactions to uncover who our customers really are, how and when they spend their money, and where the biggest levers for revenue growth lie.*
  > 
  > *Over the next five minutes, I will walk you through our complete analytical lifecycle—from baseline data hygiene and statistical profiling to non-obvious behavioral paradoxes and concrete, high-ROI business recommendations."*

---

### Slide 2: Dataset Architecture & Data Hygiene
* **Estimated Time:** 0:30 – 1:00 (30 seconds)
* **What to Show on Screen:**
  - Slide 2 showing metrics: 1,000 Records, 9 Features, 0 Missing Values, 0 Duplicates.
  - Visualization: [01_numerical_distributions_hist_kde.png](screenshots/01_numerical_distributions_hist_kde.png) (Distributions & KDE).
* **What to Speak (Script):**
  > *"Let's begin with our foundation: dataset architecture and data hygiene.*
  > 
  > *We analyzed 1,000 customer transactions spanning an entire calendar year from January 2023 through January 2024 across nine features.*
  > 
  > *The first major finding was the pristine quality of the data: zero null or missing values, zero duplicated rows, and 100% unique transaction IDs. Our catalog spans three core retail departments—Clothing, Electronics, and Beauty—with customer ages ranging from 18 to 64 years old. Having such high data hygiene allowed us to proceed directly into exploratory analysis without risk of imputation bias."*

---

### Slide 3: Descriptive Statistics & Revenue Skewness
* **Estimated Time:** 1:00 – 1:35 (35 seconds)
* **What to Show on Screen:**
  - Slide 3 showing $456.00 Mean, $135.00 Median, +1.38 Skewness.
  - Visualization: [02_numerical_boxplots.png](screenshots/02_numerical_boxplots.png) (Boxplots showing IQR spreads).
* **What to Speak (Script):**
  > *"Moving to our descriptive statistics, we uncover a crucial structural finding: extreme right-skewed revenue distribution.*
  > 
  > *While fifty percent of our transactions are under $135—which is our median spend—the average order value is pulled up more than three-fold to $456.00, with a positive skewness of +1.38.*
  > 
  > *Why does this happen? Because our product catalog is bifurcated into discrete price tiers: everyday essentials between $25 and $50, and premium high-margin goods at $300 and $500. Meanwhile, customer age and order quantities display near-perfect symmetrical distributions with zero abnormal outliers."*

---

### Slide 4: Temporal Seasonality & Monthly Velocity
* **Estimated Time:** 1:35 – 2:10 (35 seconds)
* **What to Show on Screen:**
  - Slide 4 showing Monthly Trends: May Peak ($53,150), September Trough ($23,620), Q4 Holiday Surge ($126,190).
  - Visualization: [03_monthly_sales_trend.png](screenshots/03_monthly_sales_trend.png) (Monthly line chart with 2023 baseline).
* **What to Speak (Script):**
  > *"Next, we conducted a time-series analysis to model sales velocity across the year. The business follows a pronounced bimodal seasonal rhythm.*
  > 
  > *Our top-performing month was May, reaching over $53,000 in revenue across 105 transactions. Following the spring surge, we observed a late-summer trough in September at $23,600.*
  > 
  > *However, this is immediately followed by a massive 97% rebound in October, which kicks off a powerful fourth-quarter holiday acceleration generating over $126,000. Interestingly, throughout these volume fluctuations, our Average Order Value remained remarkably stable at approximately $450 to $500."*

---

### Slide 5: Day-of-Week Shopping Velocity
* **Estimated Time:** 2:10 – 2:40 (30 seconds)
* **What to Show on Screen:**
  - Slide 5 showing Saturday Sales ($78,815 across 150 orders) vs Thursday Trough ($53,835).
  - Visualization: [06_day_of_week_sales_velocity.png](screenshots/06_day_of_week_sales_velocity.png).
* **What to Speak (Script):**
  > *"When we analyze daily shopping patterns throughout the week, Saturday is the undisputed champion, generating nearly $79,000 across 150 transactions.*
  > 
  > *Midweek days—particularly Wednesdays and Thursdays—represent the lowest velocity, dipping to around $54,000.*
  > 
  > *This 46% surge between Thursday and Saturday gives our operations and digital marketing teams clear guidance on exactly when to deploy staff and concentrate ad spend."*

---

### Slide 6: Customer Demographics & Gender Parity
* **Estimated Time:** 2:40 – 3:15 (35 seconds)
* **What to Show on Screen:**
  - Slide 6 showing 51.0% Female vs 49.0% Male Spenders; $456.55 Female AOV vs $455.43 Male AOV.
  - Visualization: [08_gender_demographics_and_spending.png](screenshots/08_gender_demographics_and_spending.png).
* **What to Speak (Script):**
  > *"Turning to customer demographics, our data revealed extraordinary parity.*
  > 
  > *Female shoppers contribute 51% of revenue while male shoppers contribute 49%. What's truly remarkable is that their Average Order Value is virtually identical: $456.55 for females versus $455.43 for males—a difference of barely one dollar.*
  > 
  > *Furthermore, spending power is distributed evenly across all age cohorts, from young adults to mature seniors. This completely invalidates any assumption that our store appeals solely to a specific age bracket or gender."*

---

### Slide 7: Product Category Performance
* **Estimated Time:** 3:15 – 3:45 (30 seconds)
* **What to Show on Screen:**
  - Slide 7 showing Electronics ($156.6k, 34.3%), Clothing ($155.6k, 34.1%), Beauty ($143.8k, 31.5%).
  - Visualization: [11_product_category_performance.png](screenshots/11_product_category_performance.png).
* **What to Speak (Script):**
  > *"Looking at product performance across our three departments, we see a remarkably balanced portfolio.*
  > 
  > *Electronics leads slightly at 34.3% of total revenue, Clothing follows closely at 34.1%, and Beauty captures 31.5%.*
  > 
  > *Because the spread between our highest and lowest department is less than 3%, our business is well-hedged against category-specific supply chain disruptions or consumer demand shifts."*

---

### Slide 8: Catalog Price Tiers & Revenue Domination
* **Estimated Time:** 3:45 – 4:15 (30 seconds)
* **What to Show on Screen:**
  - Slide 8 showing the 88/40 Rule (Items at $300 & $500 drive 88.4% of revenue from 39.6% volume).
  - Visualization: [12_top_selling_products_and_price_tiers.png](screenshots/12_top_selling_products_and_price_tiers.png).
* **What to Speak (Script):**
  > *"When we analyze individual catalog price points, we uncover an eye-opening 88-40 rule.*
  > 
  > *Items priced in our top tiers—at $300 and $500—account for less than 40% of unit volume, yet they generate over 88% of our total company revenue!*
  > 
  > *Our entry-level products priced between $25 and $50 drive volume and foot traffic, but our financial lifeblood is overwhelmingly powered by top-tier premium basket conversions."*

---

### Slide 9: Correlation Dynamics & Basket Inelasticity
* **Estimated Time:** 4:15 – 4:45 (30 seconds)
* **What to Show on Screen:**
  - Slide 9 showing Pearson correlation matrix: r = +0.85 (Price vs Total Spend), r = +0.017 (Price vs Quantity).
  - Visualization: [13_numerical_correlation_heatmaps.png](screenshots/13_numerical_correlation_heatmaps.png).
* **What to Speak (Script):**
  > *"Our correlation heatmaps reveal two vital statistical insights.*
  > 
  > *First, unit price is the dominant driver of transaction spend at r = 0.85.*
  > 
  > *Second, and most surprisingly, the correlation between unit price and basket quantity is practically zero at plus 0.017. Shoppers do not reduce unit counts when buying expensive items; they consistently purchase between two and three items regardless of whether an item costs $25 or $500. This indicates rigid basket size inelasticity."*

---

### Slide 10: Non-Obvious Insight #1 — The Pareto Whale Economy
* **Estimated Time:** 4:45 – 5:20 (35 seconds)
* **What to Show on Screen:**
  - Slide 10 showing Lorenz Curve, Gini Coefficient of 0.6091, Top 20% = 62.89% of revenue ($286,800).
  - Visualization: [14_pareto_lorenz_spending_concentration.png](screenshots/14_pareto_lorenz_spending_concentration.png).
* **What to Speak (Script):**
  > *"Now for our advanced behavioral findings. In Day 6, we plotted the empirical Lorenz curve and measured a Gini coefficient of 0.6091.*
  > 
  > *This demonstrates severe revenue concentration: the top 20% of customer orders generate nearly 63% of our entire turnover—totaling almost $287,000 with an average order value of $1,420.*
  > 
  > *By contrast, the bottom 40% of orders generate less than 7% of revenue. Retaining and delighting this top quintile of high-value shoppers is our single highest financial priority."*

---

### Slide 11: Non-Obvious Insights #2 & #3 — Temporal Decoupling & The Young Adult Paradox
* **Estimated Time:** 5:20 – 5:55 (35 seconds)
* **What to Show on Screen:**
  - Slide 11 showing Weekend Tech Lift (+23%), Beauty peaks on Mon/Fri, and Young Adult VIP conversion (25.44%).
  - Visualization: [16_demographic_persona_archetypes_and_vip_propensity.png](screenshots/16_demographic_persona_archetypes_and_vip_propensity.png) and [15_day_of_week_category_dynamics_and_weekend_lift.png](screenshots/15_day_of_week_category_dynamics_and_weekend_lift.png).
* **What to Speak (Script):**
  > *"We also discovered two behavioral paradoxes that defy conventional retail wisdom.*
  > 
  > *First, departmental rhythms decouple: Beauty spending surges on Mondays and Fridays for personal care planning, while Electronics spending spikes by 23% on weekends.*
  > 
  > *Second, our persona analysis revealed that 18 to 25 year-olds have the highest VIP order conversion rate in the entire business at 25.4%, driven by aspirational luxury purchases. Additionally, young men purchasing beauty items spend an incredible $637 on average—over 50% more than females—representing a prime luxury gifting segment."*

---

### Slide 12: Strategic Recommendations & Business Impact
* **Estimated Time:** 5:55 – 6:30 (35 seconds)
* **What to Show on Screen:**
  - Slide 12 showing the 4 Strategic Initiatives:
    1. VIP Concierge & Whale Retention Engine (+$14,300 margin)
    2. Dayparted Marketing & Staffing (12–18% CAC reduction)
    3. Cross-Department Bundles (+$89,000 potential lift)
    4. Male Gifting & BNPL Financing (10–15% conversion lift)
* **What to Speak (Script):**
  > *"To convert these findings into bottom-line financial returns, we recommend four targeted initiatives:*
  > 
  > *First, establish an invite-only VIP Concierge program for our top-tier spenders, protecting an estimated $14,000 in annual recurring gross margin.*
  > 
  > *Second, align media schedules and staffing to day-of-week rhythms, reducing customer acquisition costs by up to 18%.*
  > 
  > *Third, introduce cross-department bundles to break through the 2.5 unit cap, unlocking up to $89,000 in incremental revenue without needing new customer traffic.*
  > 
  > *And fourth, implement Buy Now Pay Later financing and curated luxury gifting guides to capture high-margin young adult demand.*
  > 
  > *Thank you very much for your time, and I am now open to any questions!"*

---

## 🤖 Ready-to-Use AI Prompts

If you prefer to use AI tools to generate presentation slides, voiceovers, or avatar videos, use these copy-paste prompts:

### Prompt 1: For AI Slide Deck Makers (Gamma.app / Canva / Beautiful.ai)
```text
Create a modern, professional, 12-slide executive presentation for an Exploratory Data Analysis (EDA) on Retail Sales.

Audience: Retail executives, data science managers, and investors.
Tone: Data-driven, analytical, strategic, executive.
Color Palette: Slate navy (#0f172a), dark charcoal, electric cyan (#38bdf8), violet (#a855f7), and emerald green (#10b981).

Slide Structure:
Slide 1: Title Slide - "Retail Sales Exploratory Data Analysis" ($456K Gross Sales, 1,000 Transactions, 0 Missing Values, Gini 0.6091).
Slide 2: Data Architecture & Hygiene (1,000 rows, 9 features, 0 nulls, 0 duplicates, 12-month calendar horizon).
Slide 3: Descriptive Statistics & Revenue Skewness (Mean $456 vs Median $135, positive skew +1.38, 5 discrete catalog price tiers).
Slide 4: Time Series & Seasonality (May annual peak $53.1K, September slump $23.6K, October +97% rebound, $126K Q4 holiday surge).
Slide 5: Day of Week Dynamics (Saturday sales leader $78.8K across 150 orders; Thursday trough $53.8K).
Slide 6: Demographics & Gender Parity (51% Female vs 49% Male; identical AOV $456.55 vs $455.43; uniform age cohort power).
Slide 7: Product Category Performance (Balanced mix: Electronics 34.3%, Clothing 34.1%, Beauty 31.5%).
Slide 8: Catalog Pricing Structure & 88/40 Rule ($300-$500 items generate 88.4% of revenue from only 39.6% unit volume).
Slide 9: Correlation Matrix & Basket Inelasticity (Price drives spend r=+0.85; Price vs Quantity r=+0.017; customers buy ~2.5 items regardless of price).
Slide 10: Non-Obvious Insight #1 - The Pareto Whale Economy (Lorenz Curve, Gini 0.6091, top 20% of transactions generate 62.89% of revenue with $1,420 AOV).
Slide 11: Non-Obvious Insights #2 & #3 - Temporal Decoupling & Persona Paradoxes (Weekend electronics +23% lift; Young adults lead in VIP conversion at 25.4%; Young male beauty gifting AOV $637).
Slide 12: Strategic Recommendations & ROI (VIP Concierge Program, Dayparted Ad scheduling, Cross-department bundling for +$89K lift, BNPL financing).
```

### Prompt 2: For AI Voiceover Generators (ElevenLabs / Murf.ai / Speechify)
```text
Voice: Professional, warm, confident, articulate executive presenter.
Pacing: Moderate (130-145 words per minute), clear enunciation of numbers and statistics.

Script:
"Hello and welcome to this executive presentation on our Retail Sales Exploratory Data Analysis. Today, we look beyond the surface of 1,000 customer transactions to uncover fundamental purchasing behaviors and actionable growth opportunities.

Across an entire calendar year, our dataset demonstrates flawless data hygiene with zero missing values. Examining descriptive statistics reveals a significant revenue skew: while fifty percent of transactions are under $135, the mathematical average reaches $456, driven by high-ticket catalog pricing.

Temporally, the business experiences a strong bimodal pattern, peaking in May at $53,000, dipping in late summer, and rebounding by 97 percent into a $126,000 fourth-quarter holiday surge. Saturdays dominate the week, generating 46 percent higher revenue than midweek lows.

Demographically, we uncovered remarkable commercial parity: male and female shoppers spend with virtually identical average order values of $456. Furthermore, our product portfolio is evenly balanced across Electronics, Clothing, and Beauty.

Crucially, our advanced analysis revealed a severe Pareto concentration: the top 20 percent of orders drive nearly 63 percent of total enterprise turnover, with a Gini inequality index of 0.61. In addition, young adults lead the company in VIP order conversion at 25 percent, while young males buying beauty items spend an extraordinary $637 on average in luxury gifting.

To capitalize on these findings, we recommend four high-impact initiatives: establishing a VIP concierge retention program, synchronizing dayparted advertising and staffing, introducing multi-item cross-department bundles to break through the 2.5 unit basket cap for up to $89,000 in incremental sales, and deploying Buy Now Pay Later financing. Thank you for your time!"
```

### Prompt 3: For AI Avatar / Video Generators (HeyGen / Synthesia / D-ID)
```text
Avatar: Professional business analyst in modern business casual attire against a clean corporate studio or tech office background.
Framing: Medium close-up with side-by-side presentation screen.
Emotion: Confident, engaging, trustworthy.
Language: English (US / Global).
Input Script: [Use the Voiceover Script from Prompt 2 above].
```

### Prompt 4: Video Title, Description & YouTube/Portfolio Metadata
```text
TITLE:
Retail Sales EDA: End-to-End Data Analysis, Hidden Trends & Business Strategy | Python & Pandas

DESCRIPTION:
An executive walkthrough of our comprehensive Exploratory Data Analysis (EDA) on retail sales performance across 1,000 customer transactions ($456,000 in gross turnover).

📊 Key Findings Covered:
- Complete data hygiene audit & descriptive statistical profiling
- Bimodal seasonality (May peak, late-summer slump, Q4 holiday acceleration)
- Customer demographic parity across age cohorts and gender ($456 AOV parity)
- The 88/40 Catalog Rule: high-ticket items driving 88.4% of total turnover
- Non-Obvious Insight: The Pareto Whale Economy (Gini 0.6091, Top 20% = 62.9% revenue)
- Persona Paradox: Young adults leading in VIP order conversion rate (25.4%)
- 4 Data-Backed Strategic Recommendations with projected ROI lifts

⏱️ Timestamps:
0:00 - Introduction & Project Scope
0:30 - Data Architecture & Hygiene Audit
1:00 - Descriptive Statistics & Spending Skewness
1:35 - Time Series & Monthly Seasonality
2:10 - Day-of-Week Shopping Velocity
2:40 - Demographics & Gender Parity Analysis
3:15 - Product Category Portfolio Balance
3:45 - Catalog Pricing Tiers & 88/40 Rule
4:15 - Correlation Matrix & Basket Inelasticity
4:45 - Non-Obvious Insight #1: The Pareto Whale Economy (Gini 0.61)
5:20 - Non-Obvious Insights #2 & #3: Weekend Lift & Young Adult Paradox
5:55 - Strategic Recommendations & Revenue Projections
6:30 - Conclusion

🔗 GitHub Repository: https://github.com/addu91577-sketch/OIBSIP/tree/main/DataAnalytics-L1-EDARetailSales
💻 Tech Stack: Python 3, Pandas, Matplotlib, Seaborn, Jupyter Notebook
```

---

## 🎙️ Recording Tips for a High-Scoring Presentation

1. **Screen Resolution:** Record in **1920×1080 (1080p)** or 1440p. Press `F11` in your browser when viewing `presentation/index.html` to eliminate browser tab clutter.
2. **Audio Quality:** Use a headset or external microphone if available. Speak with clear, steady pacing (around 130–145 words per minute).
3. **Cursor Cues:** Use your mouse cursor purposefully to circle key numbers (e.g. circle the $456 AOV, or the Gini 0.6091) as you mention them.
4. **Energy & Confidence:** Keep an enthusiastic, executive tone. Frame every chart not just as "here is a chart", but as *"what this chart means for the business"*.
