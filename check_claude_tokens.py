import json

# Check the successful Claude 30-day run
with open(
    "results/json/claude-3-haiku-20240307_1games_30days_v05_20250829_120404.json"
) as f:
    data = json.load(f)

game = data["results"]["claude-3-haiku-20240307"]["individual_games"][0]
tokens = game["token_usage"]

print("Claude Haiku (successful 30-day run):")
print(f"  Input tokens: {tokens['input_tokens']:,}")
print(f"  Cached tokens: {tokens['cached_input_tokens']:,}")
print(f"  Output tokens: {tokens['output_tokens']:,}")
print(f"  Total tokens: {tokens['total_tokens']:,}")
print(f"  Days played: {game['days_played']}")
print(f"  Total profit: ${float(game['total_profit']):.2f}")
