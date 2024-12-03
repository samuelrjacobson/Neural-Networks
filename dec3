# ----------------------- Imports -----------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime  # For handling dates

# For data fetching (ensure cfbd is installed and configured properly)
from cfbd import Configuration, ApiClient, TeamsApi, GamesApi, StatsApi, RecruitingApi

# For model building
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# For preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ----------------------- Data Fetching and Preprocessing -----------------------

# Securely load the API key
configuration = Configuration()
configuration.api_key['Authorization'] = 'PIX1Z54E93iZiIa+ZPDwFxhiI+pcS+ZsOjQYcPwIUI9dWJDf5/ur3C+X21aXgKVq'
configuration.api_key_prefix['Authorization'] = 'Bearer'

api_client = ApiClient(configuration)
teams_api = TeamsApi(api_client)
games_api = GamesApi(api_client)
stats_api = StatsApi(api_client)
recruiting_api = RecruitingApi(api_client)

# Teams and years of interest
teams = [
    "Michigan", "Michigan State", "Ohio State", "Penn State", "Maryland",
    "Indiana", "Rutgers", "Northwestern", "UCLA", "Iowa", "Minnesota", "USC",
    "Nebraska", "Purdue", "Washington", "Oregon", "Wisconsin", "Illinois", "Eastern Michigan"
]
years = [2024, 2023, 2022, 2021, 2020, 2019, 2018]  # Adjust years as needed

def get_recruit_rank(team, year):
    team_recruit_stats = recruiting_api.get_recruiting_teams(year=year, team=team)
    team_recruit_rank = None
    if team_recruit_stats:
        team_recruit_rank = team_recruit_stats[0].rank  # Store the first rank (only one rank per team per year)
    return team_recruit_rank

# Fetch seasonal statistics for each team
team_season_stats_list = []
for year in years:
    for team in teams:
        try:
            # Extract recruiting rank
            rank = get_recruit_rank(team, year)

            # Get stats from stats api
            team_season_stats = stats_api.get_team_season_stats(year=year, team=team)

            # Create a dictionary to hold the stats for easy access
            stats_dict = {}
            for stat in team_season_stats:
                stats_dict[stat.stat_name] = stat.stat_value

            # Extract necessary stats with defaults if not present
            passAttempts = stats_dict.get('passAttempts', 0)
            passCompletions = stats_dict.get('passCompletions', 0)

            passCompPercent = (
                0 if passAttempts == 0 else passCompletions / passAttempts
            )

            season_stats_dict = {
                'team': team,
                'year': year,
                'recruitRank': rank,
                'passCompPercent': passCompPercent,
                'passingTDs': stats_dict.get('passingTDs', 0),
                'rushingTDs': stats_dict.get('rushingTDs', 0),
                'penaltyYards': stats_dict.get('penaltyYards', 0),
                'fumblesLost': stats_dict.get('fumblesLost', 0),
                'interceptions': stats_dict.get('interceptions', 0),
                'fumblesRecovered': stats_dict.get('fumblesRecovered', 0),
                'passesIntercepted': stats_dict.get('passesIntercepted', 0),
                'pointsPerGame': stats_dict.get('pointsPerGame', 0),
                'yardsPerGame': stats_dict.get('yardsPerGame', 0)
            }

            team_season_stats_list.append(season_stats_dict)

        except Exception as e:
            print(f"Error fetching season stats for {team} in {year}: {e}")

# Convert seasonal stats to DataFrame
df_season_stats = pd.DataFrame(team_season_stats_list)

# Fetch game-specific data
team_game_stats_list = []
for year in years:
    for team in teams:
        try:
            response = games_api.get_games(year=year, team=team, season_type='regular')
            games = response

            for game in games:
                # Determine if the game has been played
                game_played = game.home_points is not None and game.away_points is not None

                if game_played:
                    if game.home_team == team:
                        opponent = game.away_team
                        points_for = game.home_points
                        points_against = game.away_points
                        is_home = 1
                    else:
                        opponent = game.home_team
                        points_for = game.away_points
                        points_against = game.home_points
                        is_home = 0

                    game_stats_dict = {
                        'team': team,
                        'opponent': opponent,
                        'points_for': points_for,
                        'points_against': points_against,
                        'year': year,
                        'week': game.week,
                        'is_home': is_home
                    }
                    team_game_stats_list.append(game_stats_dict)
                else:
                    # For upcoming games without scores, you can choose to include them or not
                    # Here, we'll skip them as they are not part of the training data
                    continue

        except Exception as e:
            print(f"Error fetching data for {team} in {year}: {e}")

# Convert game-specific data to DataFrame
df_game_stats = pd.DataFrame(team_game_stats_list)

# Merge game data with seasonal stats
df_combined = df_game_stats.merge(df_season_stats, on=['team', 'year'], how='left')

# Preprocess data
df_combined.fillna(0, inplace=True)

# Define target variables
targets = df_combined[['points_for', 'points_against']]

# Define feature columns
feature_columns = [
    'team', 'opponent', 'year', 'week', 'is_home', 'recruitRank', 'passCompPercent', 'passingTDs', 'rushingTDs',
    'penaltyYards', 'fumblesLost', 'interceptions', 'fumblesRecovered', 'passesIntercepted',
    'pointsPerGame', 'yardsPerGame'
]
features = df_combined[feature_columns]

# One-hot encode categorical features
features_encoded = pd.get_dummies(features, columns=['team', 'opponent'], drop_first=True)

# Standardize numerical features
numerical_features = [
    'year', 'week', 'recruitRank', 'passCompPercent', 'passingTDs', 'rushingTDs', 'penaltyYards',
    'fumblesLost', 'interceptions', 'fumblesRecovered', 'passesIntercepted',
    'pointsPerGame', 'yardsPerGame'
]
scaler_features = StandardScaler()
features_encoded[numerical_features] = scaler_features.fit_transform(features_encoded[numerical_features])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_encoded, targets, test_size=0.2, random_state=42
)

# ----------------------- Neural Network Training -----------------------

# Convert targets to numpy arrays
y_train = y_train.values
y_test = y_test.values

#gpt
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


# Define the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(2)  # Output layer for regression
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    verbose=1
)

# ----------------------- Model Evaluation -----------------------

# Evaluate the model on the test set
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {test_mae}")

# Make predictions
predictions = model.predict(X_test)

# Calculate additional metrics
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Test MSE: {mse}")
print(f"Test MAE: {mae}")
print(f"R² Score: {r2}")

# Plot training & validation loss values
# plt.figure(figsize=(12, 4))

# Loss
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Train')
# plt.plot(history.history['val_loss'], label='Validation')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss (MSE)')
# plt.legend()

# MAE
# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='Train')
# plt.plot(history.history['val_mae'], label='Validation')
# plt.title('Model MAE')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()

# plt.show()

# ----------------------- Predictions vs Actual -----------------------

# Creates a DataFrame to compare predictions and actual values
# Ensure X_test is a DataFrame with proper columns
X_test = pd.DataFrame(X_test, columns=features_encoded.columns)

# Now apply the filter method to extract the team columns
comparison_df = pd.DataFrame({
    'Team': X_test.filter(regex='^team_').idxmax(axis=1).str.replace('team_', ''),
    'Opponent': X_test.filter(regex='^opponent_').idxmax(axis=1).str.replace('opponent_', ''),
    'Predicted Points For': predictions[:, 0],
    'Predicted Points Against': predictions[:, 1],
    'Actual Points For': y_test[:, 0],
    'Actual Points Against': y_test[:, 1]
})


print("\nPredictions vs Actual:")
print(comparison_df.head(10))


# ----------------------- Projection Functionality -----------------------

def project_game(home_team, away_team, year, week, scaler_features, model, features_encoded, df_season_stats):
    # Retrieve the home team's recruiting rank
    home_team_recruit_rank = get_recruit_rank(home_team, year)
    
    # Retrieve home team stats for the given year
    team_stats = df_season_stats[(df_season_stats['team'] == home_team) & (df_season_stats['year'] == year)]
    if team_stats.empty:
        print(f"No stats found for team {home_team} in year {year}. Using zeros for stats.")
        stats = {key: 0 for key in numerical_features}
    else:
        stats = team_stats.iloc[0].to_dict()
        # Extract only the required stats
        stats = {k: stats[k] for k in numerical_features if k in stats}

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'team': [home_team],
        'opponent': [away_team],
        'year': [year],
        'week': [week],
        'is_home': [1],
        'recruitRank': [home_team_recruit_rank],
        'passCompPercent': [stats.get('passCompPercent', 0)],
        'passingTDs': [stats.get('passingTDs', 0)],
        'rushingTDs': [stats.get('rushingTDs', 0)],
        'penaltyYards': [stats.get('penaltyYards', 0)],
        'fumblesLost': [stats.get('fumblesLost', 0)],
        'interceptions': [stats.get('interceptions', 0)],
        'fumblesRecovered': [stats.get('fumblesRecovered', 0)],
        'passesIntercepted': [stats.get('passesIntercepted', 0)],
        'pointsPerGame': [stats.get('pointsPerGame', 0)],
        'yardsPerGame': [stats.get('yardsPerGame', 0)]
    })

    # One-Hot Encode 'team' and 'opponent' columns
    input_encoded = pd.get_dummies(input_data, columns=['team', 'opponent'], drop_first=True)

    # Align the input_encoded DataFrame to the trained features_encoded
    input_encoded = input_encoded.reindex(columns=features_encoded.columns, fill_value=0)

    # Standardize numerical features
    input_encoded[numerical_features] = scaler_features.transform(input_encoded[numerical_features])

    # Make prediction
    predicted_output = model.predict(input_encoded)

    # Extract projected scores
    projected_home = predicted_output[0][0]
    projected_away = predicted_output[0][1]

    return projected_home, projected_away


def get_available_teams(features_encoded):
    team_columns = [col for col in features_encoded.columns if col.startswith('team_')]
    available_teams = sorted([col.replace('team_', '') for col in team_columns])
    return available_teams


available_teams = get_available_teams(features_encoded)


def get_user_input(available_teams, years):
    print("\n=== Project a Game Score ===")
    print("Available Teams:")
    for team in available_teams:
        print(f"- {team}")

    # Input Home Team
    while True:
        home_team = input("\nEnter the Home Team: ").strip()
        if home_team in available_teams:
            break
        else:
            print("Invalid team name. Please choose from the available teams listed above.")

    # Input Away Team
    while True:
        away_team = input("Enter the Away Team: ").strip()
        if away_team in available_teams:
            break
        else:
            print("Invalid team name. Please choose from the available teams listed above.")

    # Input Year
    while True:
        try:
            year = int(input("Enter the Year (e.g., 2023): ").strip())
            if year in years:
                break
            else:
                print(f"Year not in the dataset. Available years: {years}")
        except ValueError:
            print("Please enter a valid year as an integer.")

    # Input Week
    while True:
        try:
            week = int(input("Enter the Week (e.g., 1): ").strip())
            if 1 <= week <= 15:  # Assuming weeks range from 1 to 15
                break
            else:
                print("Week should be between 1 and 15.")
        except ValueError:
            print("Please enter a valid week number as an integer.")

    return home_team, away_team, year, week


def display_projected_score(home_team, away_team, projected_home, projected_away, overtime, prediction_date):
    print("\n=== Projected Game Score ===")
    print(f"Date of Prediction: {prediction_date}")
    print(f"{home_team} (Home) vs {away_team} (Away)")
    if overtime == False:
        print(f"Projected Score:\n{home_team}: {projected_home:.2f}\n{away_team}: {projected_away:.2f}")
    else:
        print(f"Projected Score:\n{home_team}: {projected_home:.2f}\n{away_team}: {projected_away:.2f} (OT)")

def display_projected_score_no_date(home_team, away_team, projected_home, projected_away, overtime, week):
    print("\n=== Projected Game Score ===")
    print(f"Week: {week}")
    print(f"{home_team} (Home) vs {away_team} (Away)")
    if overtime == False:
        print(f"Projected Score:\n{home_team}: {projected_home:.2f}\n{away_team}: {projected_away:.2f}")
    else:
        print(f"Projected Score:\n{home_team}: {projected_home:.2f}\n{away_team}: {projected_away:.2f} (OT)")

def clean_scores(projected_home, projected_away):
    home_score = int(projected_home)
    away_score = int(projected_away)
    overtime = False

    # Remove impossible scores
    if home_score == 1 or home_score < 0:
        home_score = 0
    if away_score == 1 or away_score < 0:
        away_score = 0

    # If score is tied, go to overtime
    if home_score == away_score:
        overtime = True
        if projected_home > projected_home:
            home_score += 2
        else:
            away_score += 2

    return home_score, away_score, overtime

def main_projection():
    home_team, away_team, year, week = get_user_input(available_teams, years)
    projected_home, projected_away = project_game(
        home_team, away_team, year, week,
        scaler_features, model, features_encoded, df_season_stats
    )
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    home_score, away_score, overtime = clean_scores(projected_home, projected_away)
    display_projected_score(home_team, away_team, home_score, away_score, overtime, current_date)

# ----------------------- New Feature: Next Week's Matchups Prediction -----------------------

def estimate_current_week(current_date, season_start_date=datetime.datetime(2024, 9, 1)):
    
    delta_days = (current_date - season_start_date).days
    estimated_week = delta_days // 7 + 1
    if estimated_week < 1:
        estimated_week = 1
    elif estimated_week > 15:
        estimated_week = 15
    return estimated_week


def fetch_next_week_matchups(api_client, teams, current_year, next_week):

    current_matchups = []
    try:
        # Fetch all games for the current year and next week
        response = games_api.get_games(year=current_year, week=next_week, season_type='regular')
        games = response

        for game in games:
            home_team = game.home_team
            away_team = game.away_team

            # Only consider matchups where both teams are in the specified teams list
            if home_team in teams and away_team in teams:
                matchup = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'year': current_year,
                    'week': next_week
                }
                current_matchups.append(matchup)
    except Exception as e:
        print(f"Error fetching next week matchups: {e}")

    return current_matchups


def predict_next_week_matchups(model, scaler_features, features_encoded, df_season_stats, current_date, teams, years):

    # Estimate current week
    today = datetime.datetime.now()
    current_year = today.year
    current_week = estimate_current_week(today)
    next_week = current_week + 1
    if next_week > 15:
        next_week = 15  # Adjust based on maximum number of weeks in the season

    print(f"\n=== Next Week's Matchups Prediction (Week {next_week}, {current_year}) ===")
    print(f"Date of Prediction: {current_date}\n")

    # Fetch next week matchups
    next_week_matchups = fetch_next_week_matchups(api_client, teams, current_year, next_week)

    if not next_week_matchups:
        print("No matchups found for the next week involving the specified teams.")
        return

    # Iterate through each matchup and make predictions
    for matchup in next_week_matchups:
        home_team = matchup['home_team']
        away_team = matchup['away_team']
        year = matchup['year']
        week = matchup['week']

        projected_home, projected_away = project_game(
            home_team, away_team, year, week,
            scaler_features, model, features_encoded, df_season_stats
        )
        home_score, away_score, overtime = clean_scores(projected_home, projected_away)
        display_projected_score(home_team, away_team, home_score, away_score, overtime, current_date)


def season_projection():
    print("\n=== Project a Team's Season ===")
    print("Available Teams:")
    for team in available_teams:
        print(f"- {team}")

    # Input Team
    while True:
        chosen_team = input("\nEnter the Team: ").strip()
        if chosen_team in available_teams:
            break
        else:
            print("Invalid team name. Please choose from the available teams listed above.")
    
    # Input Year
    while True:
        try:
            year = int(input("Enter the Year (e.g., 2023): ").strip())
            if year in years:
                break
            else:
                print(f"Year not in the dataset. Available years: {years}")
        except ValueError:
            print("Please enter a valid year as an integer.")

    schedule = games_api.get_games(year=year, team=chosen_team, season_type='regular')

    for game in schedule:
        #find home team, away, week
        home_team = game.home_team
        away_team = game.away_team
        year = game.season
        week = game.week

        #predict game
        projected_home, projected_away = project_game(
            home_team, away_team, year, week,
            scaler_features, model, features_encoded, df_season_stats
        )
        home_score, away_score, overtime = clean_scores(projected_home, projected_away)
        display_projected_score_no_date(home_team, away_team, home_score, away_score, overtime, week)

# ----------------------- Execute Projection -----------------------
while True:
    user_input = input(
        "\nDo you want to (1) Project a custom game score, (2) Predict next week's matchups?, or (3) Predict a team's season? (Enter 1, 2, or 3 or 'quit' to exit): ").strip().lower()

    if user_input in ["quit", "q"]:
        print("Thank you for using the game prediction tool. Goodbye!")
        break
    elif user_input == "1":
        main_projection()
    elif user_input == "2":
        # Get current date in string format
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        predict_next_week_matchups(
            model, scaler_features, features_encoded, df_season_stats, current_date, teams, years
        )
    elif user_input == "3":
        season_projection()
    else:
        print("Invalid input. Please enter '1', '2', '3', or 'quit'.")
