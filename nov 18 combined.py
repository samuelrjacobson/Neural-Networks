import numpy as np
import pandas as pd
from cfbd import Configuration, ApiClient, TeamsApi, GamesApi, StatsApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.special import expit

# ----------------------- Data Fetching and Preprocessing -----------------------

# Securely load the API key
configuration = Configuration()
configuration.api_key['Authorization'] = 'PIX1Z54E93iZiIa+ZPDwFxhiI+pcS+ZsOjQYcPwIUI9dWJDf5/ur3C+X21aXgKVq'
configuration.api_key_prefix['Authorization'] = 'Bearer'

api_client = ApiClient(configuration)
teams_api = TeamsApi(api_client)
games_api = GamesApi(api_client)
stats_api = StatsApi(api_client)

# Teams and years of interest
teams = ["Michigan", "Michigan State", "Ohio State",
         "Penn State", "Maryland", "Indiana", "Rutgers",
         "Northwestern", "UCLA", "Iowa", "Minnesota", "USC",
         "Nebraska", "Purdue", "Washington", "Oregon", "Wisconsin",
         "Illinois"
         ]
years = [2024, 2023, 2022, 2021, 2020, 2019]

# Fetch seasonal statistics for each team
team_season_stats_list = []
for year in years:
    for team in teams:
        try:
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
                'passCompPercent': passCompPercent,
                'passingTDs': stats_dict.get('passingTDs', 0),
                'rushingTDs': stats_dict.get('rushingTDs', 0),
                'penaltyYards': stats_dict.get('penaltyYards', 0),
                'fumblesLost': stats_dict.get('fumblesLost', 0),
                'interceptions': stats_dict.get('interceptions', 0),
                'fumblesRecovered': stats_dict.get('fumblesRecovered', 0),
                'passesIntercepted': stats_dict.get('passesIntercepted', 0)
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
                # Skip games that haven't been played yet
                if game.home_points is None or game.away_points is None:
                    continue

                if game.home_team == team:
                    opponent = game.away_team
                    points_for = game.home_points
                    points_against = game.away_points
                else:
                    opponent = game.home_team
                    points_for = game.away_points
                    points_against = game.home_points

                game_stats_dict = {
                    'team': team,
                    'opponent': opponent,
                    'points_for': points_for,
                    'points_against': points_against,
                    'year': year,
                    'week': game.week
                }
                team_game_stats_list.append(game_stats_dict)
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
    'team', 'opponent', 'year', 'week', 'passCompPercent', 'passingTDs', 'rushingTDs',
    'penaltyYards', 'fumblesLost', 'interceptions', 'fumblesRecovered', 'passesIntercepted'
]
features = df_combined[feature_columns]

# Retain 'team' and 'opponent' names for output
team_opponent = df_combined[['team', 'opponent']]

# One-hot encode categorical features
features_encoded = pd.get_dummies(features, columns=['team', 'opponent'], drop_first=True)

# Standardize features
scaler_features = StandardScaler()
standardized_features = scaler_features.fit_transform(features_encoded)

# Scale target variables from 0 to 1
scaler_targets = MinMaxScaler()
scaled_targets = scaler_targets.fit_transform(targets)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, team_train, team_test = train_test_split(
    standardized_features, scaled_targets, team_opponent, test_size=0.2, random_state=42
)


# ----------------------- Neural Network Training -----------------------

# Initialize neural network weights and biases
def xavier_init(size_in, size_out):
    limit = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-limit, limit, (size_in, size_out))


input_neurons = X_train.shape[1]
hidden_neurons_1 = 10
hidden_neurons_2 = 5
output_neurons = 2
learning_rate = 0.01
epochs = 10000

np.random.seed(0)
weights_input_hidden1 = xavier_init(input_neurons, hidden_neurons_1)
weights_hidden1_hidden2 = xavier_init(hidden_neurons_1, hidden_neurons_2)
weights_hidden2_output = xavier_init(hidden_neurons_2, output_neurons)


def sigmoid(x):
    #return 1 / (1 + np.exp(-x))
    return expit(x)


def sigmoid_derivative(x):
    return x * (1 - x)


# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer1_input = np.dot(X_train, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    predicted_output = np.dot(hidden_layer2_output, weights_hidden2_output)

    # Compute error
    error = np.mean((y_train - predicted_output) ** 2)

    # Backpropagation
    output_error = y_train - predicted_output
    output_delta = output_error  # Linear activation derivative is 1

    hidden_layer2_error = np.dot(output_delta, weights_hidden2_output.T)
    hidden_layer2_delta = hidden_layer2_error * sigmoid_derivative(hidden_layer2_output)

    hidden_layer1_error = np.dot(hidden_layer2_delta, weights_hidden1_hidden2.T)
    hidden_layer1_delta = hidden_layer1_error * sigmoid_derivative(hidden_layer1_output)

    # Update weights
    weights_hidden2_output += np.dot(hidden_layer2_output.T, output_delta) * learning_rate
    weights_hidden1_hidden2 += np.dot(hidden_layer1_output.T, hidden_layer2_delta) * learning_rate
    weights_input_hidden1 += np.dot(X_train.T, hidden_layer1_delta) * learning_rate

    # Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# ----------------------- Model Evaluation -----------------------

# Testing
hidden_layer1_input_test = np.dot(X_test, weights_input_hidden1)
hidden_layer1_output_test = sigmoid(hidden_layer1_input_test)

hidden_layer2_input_test = np.dot(hidden_layer1_output_test, weights_hidden1_hidden2)
hidden_layer2_output_test = sigmoid(hidden_layer2_input_test)

predicted_output_test = np.dot(hidden_layer2_output_test, weights_hidden2_output)

predicted_output_unscaled = scaler_targets.inverse_transform(predicted_output_test)
y_test_unscaled = scaler_targets.inverse_transform(y_test)

test_error = mean_squared_error(y_test_unscaled, predicted_output_unscaled)
print(f"\nTest MSE: {test_error}\n")

print("Predictions vs Actual:")
print("Team vs Opponent | Team Predicted | Opponent Predicted | Team Actual | Opponent Actual")
print("-" * 80)
for i in range(min(len(y_test_unscaled), 10)):
    team = team_test.iloc[i]['team']
    opponent = team_test.iloc[i]['opponent']
    pred_for = predicted_output_unscaled[i][0]
    pred_against = predicted_output_unscaled[i][1]
    actual_for = y_test_unscaled[i][0]
    actual_against = y_test_unscaled[i][1]
    print(f"{team} vs {opponent} | {pred_for:.2f} | {pred_against:.2f} | {actual_for} | {actual_against}")


# ----------------------- Projection Functionality -----------------------

def project_game(home_team, away_team, year, week, scaler_features, weights_input_hidden1, weights_hidden1_hidden2,
                 weights_hidden2_output, scaler_targets, features_encoded, df_season_stats):

    # Retrieve home team stats for the given year
    team_stats = df_season_stats[(df_season_stats['team'] == home_team) & (df_season_stats['year'] == year)]
    if team_stats.empty:
        print(f"No stats found for team {home_team} in year {year}. Using zeros for stats.")
        stats = {
            'passCompPercent': 0,
            'passingTDs': 0,
            'rushingTDs': 0,
            'penaltyYards': 0,
            'fumblesLost': 0,
            'interceptions': 0,
            'fumblesRecovered': 0,
            'passesIntercepted': 0
        }
    else:
        stats = team_stats.iloc[0].to_dict()
        # Extract only the required stats
        stats = {k: stats[k] for k in ['passCompPercent', 'passingTDs', 'rushingTDs',
                                       'penaltyYards', 'fumblesLost', 'interceptions',
                                       'fumblesRecovered', 'passesIntercepted']}

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'team': [home_team],
        'opponent': [away_team],
        'year': [year],
        'week': [week],
        'passCompPercent': [stats['passCompPercent']],
        'passingTDs': [stats['passingTDs']],
        'rushingTDs': [stats['rushingTDs']],
        'penaltyYards': [stats['penaltyYards']],
        'fumblesLost': [stats['fumblesLost']],
        'interceptions': [stats['interceptions']],
        'fumblesRecovered': [stats['fumblesRecovered']],
        'passesIntercepted': [stats['passesIntercepted']]
    })

    # One-Hot Encode 'team' and 'opponent' columns
    input_encoded = pd.get_dummies(input_data, columns=['team', 'opponent'], drop_first=True)

    # Align the input_encoded DataFrame to the trained features_encoded
    # This involves ensuring all columns are present, adding missing columns with 0
    input_encoded = input_encoded.reindex(columns=features_encoded.columns, fill_value=0)

    # Standardize features
    standardized_input = scaler_features.transform(input_encoded)

    # Forward propagation
    hidden_layer1_input = np.dot(standardized_input, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    predicted_output = np.dot(hidden_layer2_output, weights_hidden2_output)
    # For regression, using linear activation in the output layer is common.

    # Since targets were scaled, inverse transform the predictions
    predicted_output_unscaled = scaler_targets.inverse_transform(predicted_output)

    # Extract projected scores
    projected_home = predicted_output_unscaled[0][0]
    projected_away = predicted_output_unscaled[0][1]

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


def display_projected_score(home_team, away_team, projected_home, projected_away):
    print("\n=== Projected Game Score ===")
    print(f"{home_team} (Home) vs {away_team} (Away)")
    print(f"Projected Score:\n{home_team}: {projected_home:.2f}\n{away_team}: {projected_away:.2f}")


def main_projection():
    home_team, away_team, year, week = get_user_input(available_teams, years)
    projected_home, projected_away = project_game(
        home_team, away_team, year, week,
        scaler_features, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output,
        scaler_targets, features_encoded, df_season_stats
    )
    display_projected_score(home_team, away_team, projected_home, projected_away)

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
            scaler_features, weights_input_hidden1, weights_hidden1_hidden2, weights_hidden2_output,
            scaler_targets, features_encoded, df_season_stats
        )
        display_projected_score(home_team, away_team, projected_home, projected_away)

# ----------------------- Execute Projection -----------------------
while True:
    user_input = input("Do you want another game prediction? (yes/no): ").strip().lower()

    if user_input == "no":
        print("Thank you for using the game prediction tool. Goodbye!")
        break
    elif user_input == "yes":
        main_projection()
    elif user_input == "season":
        season_projection()
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")