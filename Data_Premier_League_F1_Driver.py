import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from google.colab import files
import networkx as nx 

# Upload files in Google Colab
uploaded = files.upload()

drivers = pd.read_csv("drivers.csv")
driver_standings = pd.read_csv("driver_standings.csv")
constructors = pd.read_csv("constructors.csv")
constructor_results = pd.read_csv("constructor_results.csv")
constructor_standings = pd.read_csv("constructor_standings.csv")
races = pd.read_csv("races.csv")
results = pd.read_csv("results.csv")
qualifying = pd.read_csv("qualifying.csv")
circuits = pd.read_csv("circuits.csv")
pit_stops = pd.read_csv("pit_stops.csv")
lap_times = pd.read_csv("lap_times.csv")
status = pd.read_csv("status.csv")
seasons = pd.read_csv("seasons.csv")
sprint_results = pd.read_csv("sprint_results.csv")

# Function to clean and preprocess data
def preprocess_f1_data(
    races, circuits, results, drivers, constructors, driver_standings,
    constructor_standings, pit_stops, lap_times, qualifying, sprint_results
):
    races['date'] = pd.to_datetime(races['date'])
    pit_stops['time'] = pd.to_datetime(pit_stops['time'], errors='coerce')

    # Handling missing values
    driver_standings.fillna(0, inplace=True)
    constructor_standings.fillna(0, inplace=True)
    results.fillna(0, inplace=True)
    sprint_results.fillna(0, inplace=True)

    # Merging datasets
    # races with circuits
    races = races.merge(circuits, on='circuitId', how='left')

    # results with driver and constructor data
    results = results.merge(drivers, on='driverId', how='left')
    results = results.merge(constructors, on='constructorId', how='left')
    results = results.merge(races[['raceId', 'year']], on='raceId', how='left')

    # driver standings with driver information
    driver_standings = driver_standings.merge(drivers[['driverId', 'surname']], on='driverId', how='left')

    # constructor standings with constructor information
    constructor_standings = constructor_standings.merge(constructors[['constructorId', 'name']], on='constructorId', how='left')

    return {
        'races': races,
        'results': results,
        'driver_standings': driver_standings,
        'constructor_standings': constructor_standings,
        'pit_stops': pit_stops,
        'lap_times': lap_times,
        'qualifying': qualifying,
        'sprint_results': sprint_results
    }

cleaned_data = preprocess_f1_data(
    races, circuits, results, drivers, constructors, driver_standings,
    constructor_standings, pit_stops, lap_times, qualifying, sprint_results
)

for name, df in cleaned_data.items():
    df.to_csv(f'cleaned_{name}.csv', index=False)

print("\n\nData preprocessing completed! Cleaned files saved.")

#Feature Engineering
def feature_engineering():
    results_with_year = results.merge(races[['raceId', 'year']], on='raceId', how='left')
    constructor_standings_with_year = constructor_standings.merge(races[['raceId', 'year']], on='raceId', how='left')

    # ---- Driver Consistency Features ----
    driver_performance = results_with_year.groupby(['year', 'driverId']).agg(
        avg_finish_position=('positionOrder', 'mean'),
        std_finish_position=('positionOrder', 'std')
    ).reset_index()

    qualifying_performance = qualifying.groupby(['driverId']).agg(
        avg_qualifying_position=('position', 'mean')
    ).reset_index()

    driver_consistency = driver_performance.merge(qualifying_performance, on='driverId', how='left')

    # ---- Team Strength Features ----
    team_strength = constructor_standings_with_year.groupby(['year', 'constructorId']).agg(
        avg_points=('points', 'mean'),
        win_rate=('wins', lambda x: np.sum(x) / len(x))
    ).reset_index()

    # 🔹 DNF Rate per team
    dnf_status = status[status['status'].str.contains("Retired|Accident|Engine", na=False)]
    dnf_results = results_with_year[results_with_year['statusId'].isin(dnf_status['statusId'])]

    dnf_rate = dnf_results.groupby(['year', 'constructorId']).size().reset_index(name='dnf_count')
    dnf_rate = dnf_rate.merge(team_strength[['year', 'constructorId']], on=['year', 'constructorId'], how='right').fillna(0)
    dnf_rate['dnf_rate'] = dnf_rate['dnf_count'] / len(results_with_year)

    team_strength = team_strength.merge(dnf_rate[['year', 'constructorId', 'dnf_rate']], on=['year', 'constructorId'], how='left')

    # ---- Track Complexity Features ----
    lap_positions = lap_times.groupby(['raceId', 'driverId']).agg(
        position_changes=('position', lambda x: x.iloc[0] - x.iloc[-1])
    ).reset_index()

    track_overtakes = lap_positions.groupby('raceId')['position_changes'].sum().reset_index()
    track_overtakes = track_overtakes.merge(races[['raceId', 'circuitId']], on='raceId')
    track_overtakes = track_overtakes.groupby('circuitId')['position_changes'].mean().reset_index()
    track_overtakes.rename(columns={'position_changes': 'avg_overtakes'}, inplace=True)

    pit_stop_duration = pit_stops.groupby(['raceId', 'driverId']).agg(
        avg_pit_time=('milliseconds', 'mean')
    ).reset_index()

    track_pit_stops = pit_stop_duration.groupby('raceId')['avg_pit_time'].mean().reset_index()
    track_pit_stops = track_pit_stops.merge(races[['raceId', 'circuitId']], on='raceId')
    track_pit_stops = track_pit_stops.groupby('circuitId')['avg_pit_time'].mean().reset_index()

    track_complexity = track_overtakes.merge(track_pit_stops, on='circuitId', how='left')

    # Save processed data
    driver_consistency.to_csv("driver_consistency.csv", index=False)
    team_strength.to_csv("team_strength.csv", index=False)
    track_complexity.to_csv("track_complexity.csv", index=False)

    return {
        'driver_consistency': driver_consistency,
        'team_strength': team_strength,
        'track_complexity': track_complexity
    }

# 🔹 Run feature engineering
features = feature_engineering()
print("Feature engineering complete!")

#Problem Statements
#1
# Identify dominant drivers
# Calculate total races per driver
total_races = results.groupby("driverId")["raceId"].count().reset_index()
total_races.columns = ["driverId", "total_races"]

# Calculate wins and podium finishes per driver
wins = driver_standings.groupby("driverId")["wins"].sum().reset_index()
wins.columns = ["driverId", "total_wins"]

podiums = results[results["positionOrder"] <= 3].groupby("driverId")["positionOrder"].count().reset_index()
podiums.columns = ["driverId", "total_podiums"]

# Merge data
driver_performance = total_races.merge(wins, on="driverId", how="left").merge(podiums, on="driverId", how="left").fillna(0)
driver_performance = driver_performance.merge(drivers[["driverId", "forename", "surname"]], on="driverId")

driver_performance["win_ratio"] = driver_performance["total_wins"] / driver_performance["total_races"]
driver_performance["podium_ratio"] = driver_performance["total_podiums"] / driver_performance["total_races"]

dominant_drivers = driver_performance.sort_values("win_ratio", ascending=False).head(10)

# Visualization - Top Dominant Drivers
plt.figure(figsize=(12, 6))
plt.barh(dominant_drivers["forename"] + " " + dominant_drivers["surname"], dominant_drivers["win_ratio"], color="red", label="Win Ratio")
plt.barh(dominant_drivers["forename"] + " " + dominant_drivers["surname"], dominant_drivers["podium_ratio"], color="orange", alpha=0.6, label="Podium Ratio")
plt.xlabel("Ratio")
plt.ylabel("Driver")
plt.title("Top 10 Dominant Drivers Based on Win & Podium Ratios")
plt.legend()
plt.gca().invert_yaxis()
plt.show()

# Identify dominant constructors
constructor_wins = constructor_standings.groupby("constructorId")["wins"].sum().reset_index()
constructor_wins.columns = ["constructorId", "total_wins"]

constructor_podiums = results[results["positionOrder"] <= 3].groupby("constructorId")["positionOrder"].count().reset_index()
constructor_podiums.columns = ["constructorId", "total_podiums"]

constructor_races = results.groupby("constructorId")["raceId"].count().reset_index()
constructor_races.columns = ["constructorId", "total_races"]

constructor_performance = constructor_races.merge(constructor_wins, on="constructorId", how="left").merge(constructor_podiums, on="constructorId", how="left").fillna(0)
constructor_performance["win_ratio"] = constructor_performance["total_wins"] / constructor_performance["total_races"]
constructor_performance["podium_ratio"] = constructor_performance["total_podiums"] / constructor_performance["total_races"]

dominant_constructors = constructor_performance.sort_values("win_ratio", ascending=False).head(10)

# Visualization - Top Dominant Constructors
plt.figure(figsize=(12, 6))
plt.barh(dominant_constructors["constructorId"], dominant_constructors["win_ratio"], color="blue", label="Win Ratio")
plt.barh(dominant_constructors["constructorId"], dominant_constructors["podium_ratio"], color="lightblue", alpha=0.6, label="Podium Ratio")
plt.xlabel("Ratio")
plt.ylabel("Constructor ID")
plt.title("Top 10 Dominant Constructors Based on Win & Podium Ratios")
plt.legend()
plt.gca().invert_yaxis()
plt.show()

# Assessing Relationship Between Career Longevity and Success Metrics
driver_career = results.groupby("driverId")["raceId"].nunique().reset_index()
driver_career.columns = ["driverId", "career_length"]

driver_success = driver_career.merge(driver_standings.groupby("driverId")["wins"].sum().reset_index(), on="driverId")
driver_success = driver_success.merge(driver_standings.groupby("driverId")["points"].sum().reset_index(), on="driverId")

driver_success = driver_success.merge(drivers[["driverId", "forename", "surname"]], on="driverId")

# Scatter Plot - Career Longevity vs Wins
plt.figure(figsize=(12, 6))
plt.scatter(driver_success["career_length"], driver_success["wins"], c=driver_success["points"], cmap="coolwarm", s=driver_success["points"] / 10, alpha=0.7)
plt.xlabel("Career Length (Number of Races)")
plt.ylabel("Total Wins")
plt.title("Career Longevity vs Wins")
plt.colorbar(label="Total Points")
plt.show()

#2

# Merge qualifying and race results
grid_vs_race = results.merge(qualifying, on=["raceId", "driverId"], how="inner")

# Calculate position changes
grid_vs_race["position_change"] = grid_vs_race["grid"] - grid_vs_race["positionOrder"]

# Average position change per driver
driver_position_change = grid_vs_race.groupby("driverId")["position_change"].mean().reset_index()
driver_position_change = driver_position_change.merge(drivers[["driverId", "forename", "surname"]], on="driverId")

driver_position_change = driver_position_change.sort_values("position_change", ascending=False).head(25)

# Visualization - Top Drivers Gaining Positions
plt.figure(figsize=(12, 6))
plt.barh(driver_position_change["forename"] + " " + driver_position_change["surname"], driver_position_change["position_change"], color="green")
plt.xlabel("Average Positions Gained")
plt.ylabel("Driver")
plt.title("Top 20 Drivers Who Gain Most Positions in a Race")
plt.gca().invert_yaxis()
plt.show()

# Scatter plot with trendline
plt.figure(figsize=(12, 6))
plt.scatter(grid_vs_race["grid"], grid_vs_race["positionOrder"], alpha=0.5, color="purple", label="Race Results")

# Best-fit line (trendline)
z = np.polyfit(grid_vs_race["grid"], grid_vs_race["positionOrder"], 1)
p = np.poly1d(z)
plt.plot(grid_vs_race["grid"], p(grid_vs_race["grid"]), color="red", linestyle="dashed", label="Trendline")

plt.xlabel("Starting Grid Position")
plt.ylabel("Final Race Position")
plt.title("Impact of Starting Grid Position on Final Race Results")
plt.gca().invert_yaxis()  # Invert Y-axis to align with ranking order (1st is at the top)
plt.legend()
plt.show()

#3
# Print available columns for debugging
print("Pit Stops Columns:", pit_stops.columns)

# Merge with race results
pit_stops_merged = pit_stops.merge(results, on=["raceId", "driverId"], how="inner")

# Number of pit stops per driver per race
pit_stop_counts = pit_stops_merged.groupby(["raceId", "driverId"]).size().reset_index(name="pit_stop_count")

# Merge with race finishing positions
pit_stop_counts = pit_stop_counts.merge(results[["raceId", "driverId", "positionOrder"]], on=["raceId", "driverId"])

# Average number of pit stops per finishing position
avg_pit_stops = pit_stop_counts.groupby("positionOrder")["pit_stop_count"].mean().reset_index()

# Visualization 1 - Average Pit Stops per Race Finish Position
plt.figure(figsize=(12, 6))
plt.bar(avg_pit_stops["positionOrder"], avg_pit_stops["pit_stop_count"], color="red")
plt.xlabel("Final Race Position")
plt.ylabel("Average Pit Stops")
plt.title("Average Pit Stops per Race Finish Position")
plt.xticks(rotation=45)
plt.show()

# Convert 'duration' from string format ("mm:ss.sss") to float (seconds)
def convert_duration_to_seconds(duration_str):
    try:
        if isinstance(duration_str, str):
            minutes, seconds = duration_str.split(":")
            return float(minutes) * 60 + float(seconds)
        return np.nan  # Return NaN if value is not a valid string
    except:
        return np.nan  # Handle unexpected cases

# Apply conversion
pit_stops_merged["pit_stop_duration"] = pit_stops_merged["duration"].apply(convert_duration_to_seconds)

# Drop NaN values (if conversion failed for some rows)
pit_stops_merged = pit_stops_merged.dropna(subset=["pit_stop_duration"])

# Plot Heatmap - Pit Stop Duration vs. Final Race Position
plt.figure(figsize=(12, 6))

plt.hexbin(
    pit_stops_merged["pit_stop_duration"],
    pit_stops_merged["positionOrder"],
    gridsize=30, cmap="Reds", mincnt=1
)

plt.colorbar(label="Frequency")
plt.xlabel("Pit Stop Duration (Seconds)")
plt.ylabel("Final Race Position")
plt.title("Heatmap: Impact of Pit Stop Duration on Race Position")
plt.gca().invert_yaxis()  # 1st place at the top
plt.show()

#4
from itertools import combinations

# Merge results with drivers to get driver names, adding suffixes to avoid duplicates
results = results.merge(drivers[["driverId", "forename", "surname"]], on="driverId", suffixes=("", "_driver"))


# Create driver name column
results["driver_name"] = results["forename"] + " " + results["surname"]

# Function to calculate rivalries
def calculate_rivalries(results):
    head_to_head = {}

    for race_id, race_results in results.groupby("raceId"):
        driver_positions = race_results[["driver_name", "positionOrder"]].set_index("driver_name")["positionOrder"].to_dict()

        for driver1, driver2 in combinations(driver_positions.keys(), 2):
            winner, loser = (driver1, driver2) if driver_positions[driver1] < driver_positions[driver2] else (driver2, driver1)

            if (winner, loser) not in head_to_head:
                head_to_head[(winner, loser)] = 0
            head_to_head[(winner, loser)] += 1

    # Convert to DataFrame
    rivalries = pd.DataFrame([
        {"Driver1": d1, "Driver2": d2, "Competitiveness": count}
        for (d1, d2), count in head_to_head.items()
    ])

    return rivalries

# Compute rivalries
rivalries_df = calculate_rivalries(results)

# Select top 10 most competitive rivalries
top_rivalries = rivalries_df.sort_values("Competitiveness", ascending=False).head(10)

# Plot bar chart
plt.figure(figsize=(12, 6))
sns.barplot(
    data=top_rivalries,
    x="Competitiveness",
    y=top_rivalries["Driver1"] + " vs " + top_rivalries["Driver2"],
    palette="coolwarm"
)
plt.xlabel("Head-to-Head Wins")
plt.ylabel("Driver Rivalry")
plt.title("Top 10 Most Competitive F1 Rivalries (Head-to-Head)")
plt.show()

#5
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Select two drivers to swap
driver_1 = 1  # Hypothetical driver ID
team_1 = 5    # Team ID before swap

driver_2 = 2  # Hypothetical driver ID
team_2 = 10   # Team ID before swap

# Apply the swap
results.loc[results['driverId'] == driver_1, 'constructorId'] = team_2
results.loc[results['driverId'] == driver_2, 'constructorId'] = team_1

# Train a predictive model to evaluate impact
features = ["grid", "constructorId", "laps", "fastestLapSpeed", "points"]
X = results[features].dropna()
y = results["positionOrder"].dropna()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict new standings
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error after swap: {mae}")

# Visualize impact before and after swap
plt.figure(figsize=(10, 5))
sns.histplot(driver_standings['position'], bins=20, color='blue', label='Before Swap', kde=True)
sns.histplot(y_pred, bins=20, color='red', label='After Swap', kde=True)
plt.xlabel("Driver Standings Position")
plt.ylabel("Frequency")
plt.title("Impact of Driver Swap on Standings")
plt.legend()
plt.show()

#6

# Create mappings for quick lookup
driver_id_to_name = dict(zip(drivers['driverId'], drivers['surname']))
constructor_id_to_name = dict(zip(constructors['constructorId'], constructors['name']))

# Extract driver transitions
driver_transfers = results[['driverId', 'constructorId', 'raceId']].drop_duplicates()

# Plot driver movement as subgraphs
def plot_driver_movements():
    unique_drivers = driver_transfers['driverId'].unique()
    for driver_id in unique_drivers:
        driver_name = driver_id_to_name.get(driver_id, f"Driver {driver_id}")
        driver_data = driver_transfers[driver_transfers['driverId'] == driver_id].sort_values('raceId')

        G = nx.DiGraph()
        previous_team = None
        for _, row in driver_data.iterrows():
            team_name = constructor_id_to_name.get(row['constructorId'], f"Team {row['constructorId']}")
            if previous_team:
                G.add_edge(previous_team, team_name)
            previous_team = team_name

        if G.number_of_edges() > 0:  # Only plot if there are transitions
            plt.figure(figsize=(8, 5))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=10, font_weight='bold')
            plt.title(f"Driver Movements: {driver_name}")
            plt.show()

plot_driver_movements()


#7
def compare_team_performance():
    # Compute total races per team
    total_races = results.groupby('constructorId')['raceId'].nunique().reset_index()
    total_races.rename(columns={'raceId': 'total_races'}, inplace=True)

    # Compute total points per team
    team_points = constructor_standings.groupby('constructorId')['points'].sum().reset_index()
    # Merge race counts and points
    team_performance = team_points.merge(total_races, on='constructorId')
    team_performance['avg_points_per_race'] = team_performance['points'] / team_performance['total_races']
    # Compute podium finishes
    podium_finishes = results[results['positionOrder'] <= 3].groupby('constructorId')['raceId'].count().reset_index()
    podium_finishes.rename(columns={'raceId': 'podium_finishes'}, inplace=True)
    # Merge all performance metrics
    team_performance = team_performance.merge(podium_finishes, on='constructorId', how='left').fillna(0)
    team_performance = team_performance.merge(constructors[['constructorId', 'name']], on='constructorId')
    team_performance = team_performance.sort_values('avg_points_per_race', ascending=False)
    # Select only the top 25 teams
    top_teams = team_performance.head(25)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot Avg Points per Race
    sns.barplot(ax=axes[0], x='avg_points_per_race', y='name', data=top_teams, palette='coolwarm')
    axes[0].set_title("Top 25 Teams: Avg Points per Race", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Avg Points per Race", fontsize=12)
    axes[0].set_ylabel("Team", fontsize=12)
    axes[0].tick_params(axis='y', labelsize=10)  # Improve readability

    # Plot Podium Finishes
    sns.barplot(ax=axes[1], x='podium_finishes', y='name', data=top_teams, palette='coolwarm')
    axes[1].set_title("Top 25 Teams: Podium Finishes", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Podium Finishes", fontsize=12)
    axes[1].set_ylabel("Team", fontsize=12)
    axes[1].tick_params(axis='y', labelsize=10)

    plt.tight_layout()
    plt.show()

compare_team_performance()

#8
# Driver Consistency in Race Performance
def driver_consistency():
    results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')
    driver_positions = results.groupby('driverId')['positionOrder'].std().reset_index()
    driver_positions.dropna(subset=['positionOrder'], inplace=True)
    driver_positions = driver_positions.merge(drivers[['driverId', 'surname']], on='driverId', how='left')
    driver_positions.fillna("Unknown", inplace=True)
    driver_positions = driver_positions.sort_values('positionOrder', ascending=True)
    consistent_drivers = driver_positions.head(10)
    plt.figure(figsize=(12, len(consistent_drivers) * 0.5))
    sns.barplot(x='positionOrder', y='surname', data=consistent_drivers, palette='viridis')
    plt.xlabel("Standard Deviation of Race Positions", fontsize=14)
    plt.ylabel("Driver", fontsize=14)
    plt.title("Most Consistent Drivers in Race Finishes", fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()
driver_consistency()

#9
# Lap Time Efficiency
def lap_time_efficiency():
    avg_lap_times = lap_times.groupby('driverId')['milliseconds'].mean().reset_index()
    avg_lap_times = avg_lap_times.merge(drivers[['driverId', 'surname']], on='driverId')
    avg_lap_times = avg_lap_times.sort_values('milliseconds')
    plt.figure(figsize=(12, 6))
    sns.barplot(x='milliseconds', y='surname', data=avg_lap_times.head(10), palette='Blues_r')
    plt.xlabel("Average Lap Time (ms)", fontsize=14)
    plt.ylabel("Driver", fontsize=14)
    plt.title("Top 10 Fastest Drivers by Lap Time", fontsize=16)
    plt.tight_layout()
    plt.show()
lap_time_efficiency()

#10
def best_team_lineup():

    # Load datasets
    #constructors = pd.read_csv('C:\\Users\\harin\\OneDrive\\Desktop\\hackathon-data premier\\constructors.csv')
    #constructor_standings = pd.read_csv('C:\\Users\\harin\\OneDrive\\Desktop\\hackathon-data premier\\constructor_standings.csv')

    # Aggregate total points per team
    team_points = constructor_standings.groupby('constructorId')['points'].sum().reset_index()
    # Merge with team names
    team_points = team_points.merge(constructors[['constructorId', 'name']], on='constructorId')

    # Sort by points and take top 20 teams for better visualization
    top_teams = team_points.sort_values('points', ascending=False).head(20)

    # Plot with better styling
    plt.figure(figsize=(14, 8))
    sns.barplot(x='points', y='name', data=top_teams, palette='coolwarm')
    # Beautify the plot
    plt.title("Top 20 Teams Based on Total Points", fontsize=18, fontweight='bold')
    plt.xlabel("Total Team Points", fontsize=14)
    plt.ylabel("Team", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Show plot
    plt.show()
best_team_lineup()

#11
# Merge driver standings with race year
driver_trends = driver_standings.merge(races[['raceId', 'year']], on='raceId')

# Aggregate driver performance by year
driver_performance = driver_trends.groupby(['year', 'driverId']).agg(
    total_points=('points', 'sum'),
    avg_position=('position', 'mean'),
    wins=('wins', 'sum')
).reset_index()

# Get most recent data for training (last 10 years)
recent_data = driver_performance[driver_performance['year'] >= 2015]

# Prepare features and target variable
X = recent_data[['year', 'driverId']]
y = recent_data['total_points']

# One-hot encode driverId
X = pd.get_dummies(X, columns=['driverId'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Predict for 2025 season
future_year = 2025
future_drivers = driver_performance['driverId'].unique()
X_future = pd.DataFrame({'year': [future_year] * len(future_drivers), 'driverId': future_drivers})
X_future = pd.get_dummies(X_future, columns=['driverId'])

# 🔹 Ensure `X_future` has the same columns as `X_train`
missing_cols = set(X_train.columns) - set(X_future.columns)
for col in missing_cols:
    X_future[col] = 0  # Add missing columns with zero values

# 🔹 Ensure column order matches `X_train`
X_future = X_future[X_train.columns]

# Make predictions
future_predictions = model.predict(X_future)
predictions_df = pd.DataFrame({'driverId': future_drivers, 'predicted_points': future_predictions})
predictions_df = predictions_df.merge(drivers[['driverId', 'forename', 'surname']], on='driverId')

# Sort and display top predicted drivers
top_drivers = predictions_df.sort_values(by='predicted_points', ascending=False).head(10)
print("\nPredicted Top Drivers for 2025:")
print(top_drivers[['forename', 'surname', 'predicted_points']])

# Visualize Predictions
plt.figure(figsize=(10, 6))
sns.barplot(data=top_drivers, x="predicted_points", y="surname", palette="Blues_r")
plt.xlabel("Predicted Points")
plt.ylabel("Driver")
plt.title("Predicted Top 10 Drivers for 2025")
plt.show()

#12
def struggling_team_analysis():
  # Merge datasets to get season-year mapping
  standings = constructor_standings.merge(races[['raceId', 'year']], on='raceId') \
                                  .merge(constructors[['constructorId', 'name']], on='constructorId')

  # Aggregate by season
  team_trends = standings.groupby(['year', 'name']).agg(
      avg_position=('position', 'mean'),
      total_points=('points', 'sum')
  ).reset_index()

  # Identify the top struggling teams based on recent performance drop
  recent_trend = team_trends[team_trends['year'] >= 2015]

  # Perform regression to predict 2025 performance
  predictions = []
  for team in recent_trend['name'].unique():
      team_data = recent_trend[recent_trend['name'] == team].sort_values(by='year')

      if len(team_data) < 5:  # Ensure sufficient data points for prediction
          continue

      X = team_data[['year']].values
      y = team_data['avg_position'].values  # Higher position means worse performance

      model = LinearRegression()
      model.fit(X, y)

      pred_2025 = model.predict(np.array([[2025]]))[0]
      predictions.append((team, pred_2025))

  # Convert predictions to DataFrame and sort
  predictions_df = pd.DataFrame(predictions, columns=['Team', 'Predicted_Position'])
  predictions_df = predictions_df.sort_values(by='Predicted_Position', ascending=False)  # Worst teams first

  # Plot the trend
  plt.figure(figsize=(12, 6))
  sns.barplot(data=predictions_df, x='Predicted_Position', y='Team', palette='Reds_r')
  plt.xlabel("Predicted Average Standings Position (Higher = Worse)")
  plt.ylabel("Team")
  plt.title("Predicted Underperforming Teams for 2025")
  plt.show()

  print("\n**Predicted Underperforming Teams in 2025:**")
  print(predictions_df)

struggling_team_analysis()

#13
def driver_circuit_performance(driver_name):
    # Merge results with race and circuit data
    merged = results.merge(races[['raceId', 'circuitId', 'year']], on='raceId') \
                    .merge(circuits[['circuitId', 'name']], on='circuitId') \
                    .merge(drivers[['driverId', 'forename', 'surname']], on='driverId')

    # Filter for the selected driver
    merged['driver_fullname'] = merged['forename'] + " " + merged['surname']
    driver_data = merged[merged['driver_fullname'] == driver_name]

    # Calculate performance metrics
    performance = driver_data.groupby('name').agg(
        avg_position=('positionOrder', 'mean'),  # Lower is better
        win_rate=('positionOrder', lambda x: (x == 1).sum() / len(x) * 100),
        avg_points=('points', 'mean'),
        dnf_rate=('statusId', lambda x: (x > 1).sum() / len(x) * 100)  # Assumes statusId > 1 means DNF
    ).reset_index()

    # Sort by average finishing position
    performance = performance.sort_values(by='avg_position', ascending=True)

    # Plot performance at different circuits
    plt.figure(figsize=(12, 6))
    sns.barplot(data=performance, x='avg_position', y='name', palette='coolwarm')
    plt.xlabel('Average Finishing Position (Lower is Better)')
    plt.ylabel('Circuit')
    plt.title(f'Performance of {driver_name} at Different Circuits')
    plt.show()

    print("\n**Performance Metrics:**")
    print(performance)

# Example: Analyze Lewis Hamilton's track struggles
driver_circuit_performance("Lewis Hamilton")

#14
def championship_retention_probability():
    # Extract the champion (driver with the highest points) for each season
    season_champions = driver_standings.sort_values(['raceId', 'points'], ascending=[True, False]) \
                                       .groupby('raceId').first().reset_index()

    # Merge with race data to get the year
    season_champions = season_champions.merge(races[['raceId', 'year']], on='raceId', how='left')

    # Keep only relevant columns (driverId, year)
    season_champions = season_champions[['year', 'driverId']].rename(columns={'driverId': 'championId'})

    # Shift the column to check retention
    season_champions['next_season_champion'] = season_champions['championId'].shift(-1)
    season_champions['retained_title'] = season_champions['championId'] == season_champions['next_season_champion']

    # Calculate retention probability
    total_seasons = season_champions.shape[0] - 1  # Exclude the last season (no next champion)
    retained_count = season_champions['retained_title'].sum()

    retention_probability = retained_count / total_seasons if total_seasons > 0 else 0

    print(f"🔹 Historical Championship Retention Probability: {retention_probability:.2%}")

    # Display past champions with retention status
    print("\nChampionship Retention History:")
    print(season_champions[['year', 'championId', 'retained_title']])

# Call the function
championship_retention_probability()

#15
def champion_age_trends():
    # Extract the champion (driver with the highest points) for each season
    season_champions = driver_standings.sort_values(['raceId', 'points'], ascending=[True, False]) \
                                       .groupby('raceId').first().reset_index()

    # Merge with races to get the season year
    season_champions = season_champions.merge(races[['raceId', 'year']], on='raceId', how='left')

    # Merge with drivers to get birthdates
    season_champions = season_champions.merge(drivers[['driverId', 'dob']], on='driverId', how='left')

    # Convert birth date to datetime format
    season_champions['dob'] = pd.to_datetime(season_champions['dob'])

    # Calculate age at the time of winning the championship
    season_champions['champion_age'] = season_champions['year'] - season_champions['dob'].dt.year

    # Define age ranges
    bins = [20, 25, 30, 35, 40, 45, 50]
    labels = ['20-25', '26-30', '31-35', '36-40', '41-45', '46+']
    season_champions['age_group'] = pd.cut(season_champions['champion_age'], bins=bins, labels=labels, right=False)

    # Extract the decade
    season_champions['decade'] = (season_champions['year'] // 10) * 10

    # Plot
    plt.figure(figsize=(12, 6))
    sns.countplot(data=season_champions, x='age_group', hue='decade', palette="viridis")
    plt.title("Champion Age Distribution Across Decades")
    plt.xlabel("Age Group")
    plt.ylabel("Number of Championships Won")
    plt.legend(title="Decade")
    plt.show()

    # Print results
    print("\nChampion Age Distribution Over Decades:")
    print(season_champions.groupby(['decade', 'age_group'])['driverId'].count().unstack())

# Call the function
champion_age_trends()