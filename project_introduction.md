# Hong Kong Rental Price Prediction Challenge

**INTC1204 Sensing The World — Week 8 Project**

---

## What Is This Project About?

Imagine you're helping a friend find an apartment in Hong Kong. They ask: *"Is $20,000/month a fair price for a 400 sqft flat near Mong Kok?"*

You could browse listings for hours... or you could **build a model that answers this automatically**.

In this project, you will build a machine learning model to predict **monthly rental prices** for Hong Kong apartments. You'll use real transaction data, geographic information, and your creativity to make the best predictions possible.

This is how data science works in the real world — messy data, creative feature engineering, and a leaderboard to see how you stack up.

---

## How It Works

```
   You get                    You build                  You submit
 ┌──────────┐             ┌──────────────┐           ┌───────────────┐
 │ Training │  ──clean──► │ Features +   │ ──pred──► │submission.csv │
 │ Data     │  ──explore─►│ ML Model     │           │ (id + price)  │
 │ (34,528  │  ──engineer►│              │           │               │
 │  rentals)│             └──────┬───────┘           └───────┬───────┘
 └──────────┘                    │                           │
                                 │ predict                   │ upload
   Spatial Data          ┌───────▼──────┐            ┌───────▼───────┐
 ┌──────────────┐        │ Test Features│            │   Dashboard   │
 │ MTR stations │        │ (??? rows    │            │   compares    │
 │ Parks        │  ──►   │  no prices)  │            │   to hidden   │
 │ Schools      │  new   └──────────────┘            │   answers     │
 │ Malls        │  features                          └───────┬───────┘
 │ Hospitals    │  (distances, counts, etc.)                 │
 │ CBD location │                                            ▼
 └──────────────┘                                    ┌──────────────┐
                                                     │  Leaderboard │
                                                     │  (RMSE rank) │
                                                     └──────────────┘
```

**Step by step:**

1. **Explore** the training data — 34,528 real Hong Kong rental transactions with price, size, floor, district, and GPS coordinates.
2. **Engineer features** — use the provided spatial datasets (MTR stations, parks, schools, etc.) to create new columns like "distance to nearest MTR" or "number of schools within 1 km".
3. **Train a model** — try different algorithms (Random Forest, Gradient Boosting, etc.) and use cross-validation to pick the best one.
4. **Generate predictions** — apply your model to `test_features.csv` (apartments with no prices) and save a CSV with your predicted prices.
5. **Upload and check the leaderboard** — submit your CSV to the dashboard and see your RMSE score!

---

## What Data Do You Have?

### Training Data (`data/HK_house_transactions.csv`)

Each row is a real apartment rental transaction in Hong Kong.

| Column | What It Means |
|---|---|
| `area_sqft` | How big the apartment is (in square feet) |
| `floor` | What floor the apartment is on |
| `district` | Which part of Hong Kong (e.g. "Kowloon Kwun Tong District") |
| `wgs_lat`, `wgs_lon` | GPS coordinates — where the apartment is on the map |
| `Public_Housing` | Is it public housing? (True/False) |
| `price` | **Monthly rent in HKD** — this is what you're predicting! |

There are also columns for `Tower`, `Flat`, `Phase`, `Block`, and `address` — you can use them or ignore them.

### Test Features (`data/test_features.csv`)

Same columns as training data, but **without `price`**. These are the apartments you need to predict prices for.

### Spatial Datasets (for Feature Engineering)

These are **bonus datasets** with the locations of important places in Hong Kong. You can use them to calculate things like "how far is this apartment from the nearest MTR station?"

| Dataset | What's In It | How Many |
|---|---|---|
| `HK_mtr_station.csv` | MTR station locations | 372 stations |
| `HK_park.csv` | Park locations and sizes | 1,206 parks |
| `HK_school.csv` | School locations and types | 3,503 schools |
| `HK_mall.csv` | Shopping mall locations | 587 malls |
| `HK_hospital.csv` | Hospital locations | 43 hospitals |
| `HK_city_center.csv` | CBD location (Central) | 1 point |

**You can also find and use any other public Hong Kong dataset!** Bus stops, noise levels, air quality, crime data — anything you think might help predict rent.

---

## Getting Started

### 1. Open the Tutorial Notebook

Start with `w8-tutorial/w8_geovisualization_and_features.ipynb`. It walks you through:
- Making interactive maps with Folium
- Calculating distances using the Haversine formula
- Building spatial features step by step
- Testing whether a feature actually helps (using cross-validation)

### 2. Run the Starter Notebook

Open `starter_notebook.ipynb`. It gives you a working baseline model in just a few cells:
- Loads the training data and test features
- Trains a Random Forest using area, floor, and district
- Generates `my_submission.csv` ready to upload

Run it, submit the baseline, and see your first score on the leaderboard!

### 3. Improve Your Model

This is where the fun begins. Here are some ideas:

| Idea | Difficulty | Potential Impact |
|---|---|---|
| Add `dist_to_nearest_mtr` | Easy | High |
| Add `dist_to_cbd` | Easy | Medium |
| Count schools/malls within 1 km | Medium | Medium |
| Try Gradient Boosting instead of Random Forest | Medium | High |
| Tune hyperparameters with GridSearchCV | Medium | Medium |
| Use address text to extract building name | Hard | High |
| Find external data (bus stops, noise, etc.) | Hard | Could be very high! |

### 4. Submit and Iterate

Upload your prediction CSV to the dashboard (up to 5 times per day). Each time, check:
- Did your RMSE go **down**? Great, your changes helped!
- Did it go **up** or stay the same? Try a different approach.

Always validate locally with cross-validation before submitting — don't waste submissions on untested ideas.

---

## Submitting Your Predictions

### What to Upload

A CSV file with two columns:

```csv
id,price
0,18500
1,22000
2,15800
...
```

- `id` — apartment ID from `test_features.csv`
- `price` — your predicted monthly rent in HKD

Check `sample_submission.csv` for the correct format (it predicts the median for every apartment — your model should beat this!).

### Dashboard

Access the contest dashboard at: **http://113.98.54.125:8500**

Password: **ischoolisbest**

---

## How You'll Be Graded

This is **not** just about who gets the lowest RMSE. We care more about your **process** than your rank.

| Component | Weight | What We're Looking For |
|---|---|---|
| **Feature Engineering** | 40% | Did you creatively use spatial data? Did you document *why* you chose each feature? Did you try features that didn't work and explain why? |
| **Model Selection & Tuning** | 30% | Did you try multiple models? Did you use cross-validation to compare? Did you show your reasoning? |
| **Leaderboard Performance** | 20% | How does your RMSE compare to the class median? (You don't need to be #1!) |
| **Code Quality** | 10% | Is your notebook clean, readable, and reproducible? Can someone else run it? |

### What to Submit for Grading

1. Your **final prediction CSV** (uploaded to dashboard)
2. Your **code notebook** (`.ipynb`) showing your full workflow — submit via Moodle

---

## Tips from Previous Students

> *"I spent too long trying fancy models. The biggest jump in my score came from adding distance-to-MTR as a feature."*

> *"Cross-validation saved me. I thought my model was great, but CV showed it was overfitting badly."*

> *"I found a public dataset of Hong Kong bus stops and it actually improved my RMSE by $200!"*

---

## Quick Reference

| Task | Where to Look |
|---|---|
| Learn about maps and spatial features | `w8-tutorial/w8_geovisualization_and_features.ipynb` |
| Get a working baseline model | `starter_notebook.ipynb` |
| Understand the rules and data | `contest_instructions.md` |
| Check submission format | `data/sample_submission.csv` |
| Submit predictions | http://113.98.54.125:8500 |

---

## FAQ

**Q: Can I use libraries other than scikit-learn?**
Yes! You can use XGBoost, LightGBM, or any library available in a standard Python environment.

**Q: What if my model uses custom feature engineering?**
Just apply the same feature engineering to both `HK_house_transactions.csv` and `test_features.csv` before predicting. The tutorial notebook shows examples.

**Q: Can I use data from outside the provided datasets?**
Absolutely! Finding and using creative external data sources is encouraged and will be rewarded in the feature engineering grade.

**Q: What if I get a bad score?**
That's totally fine and part of the learning process. Focus on documenting what you tried, what worked, and what didn't. A well-documented notebook with a mediocre score will get a better grade than a mysterious notebook with a good score.

**Q: I'm stuck. Where do I get help?**
- Re-read the tutorial notebook — it covers the key techniques step by step
- Ask your TA during tutorial sessions
- Discuss approaches (not code!) with classmates

---

**Good luck, and have fun! This is your chance to put everything from the course together and solve a real prediction problem.**
