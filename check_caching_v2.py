import json

# Check the summary file for token usage
with open(
    "results/json/claude-3-haiku-20240307_1games_30days_v05_20250829_120404.json"
) as f:
    data = json.load(f)

result = data["results"]["claude-3-haiku-20240307"]
if "individual_games" in result and len(result["individual_games"]) > 0:
    game = result["individual_games"][0]
    tokens = game["token_usage"]

    print("Token Usage from Claude Haiku 30-day Benchmark:")
    print("=" * 60)
    print(f"Input tokens:  {tokens.get('input_tokens', 0):,}")
    print(f"Cached tokens: {tokens.get('cached_input_tokens', 0):,}")
    print(f"Output tokens: {tokens.get('output_tokens', 0):,}")
    print(f"Total tokens:  {tokens.get('total_tokens', 0):,}")

    cached = tokens.get("cached_input_tokens", 0)
    input_tokens = tokens.get("input_tokens", 0)

    if input_tokens > 0:
        cache_rate = (cached / input_tokens) * 100 if cached > 0 else 0
        print(f"\nCache hit rate: {cache_rate:.1f}%")

    print("\n" + "=" * 60)
    print("CACHING STATUS: ", end="")
    if cached > 0:
        print("✅ Prompt caching IS being used!")
    else:
        print("❌ Prompt caching is NOT being used")

    # Calculate potential savings with new pricing
    print("\n" + "=" * 60)
    print("COST ANALYSIS WITH CLAUDE 3.5 HAIKU PRICING")
    print("-" * 60)

    input_tokens = tokens.get("input_tokens", 0)
    output_tokens = tokens.get("output_tokens", 0)

    # Old pricing (what we actually paid)
    old_input_price = 0.25  # per million tokens
    old_output_price = 1.25  # per million tokens

    # New Claude 3.5 Haiku pricing
    new_input_price = 0.80  # per million tokens
    new_output_price = 4.00  # per million tokens
    new_cache_price = 0.08  # per million tokens

    # Calculate costs
    old_total = (input_tokens / 1_000_000) * old_input_price + (
        output_tokens / 1_000_000
    ) * old_output_price
    new_no_cache = (input_tokens / 1_000_000) * new_input_price + (
        output_tokens / 1_000_000
    ) * new_output_price

    # With caching (assume 85% of input can be cached after day 1)
    cached_tokens = input_tokens * 0.85
    uncached_tokens = input_tokens * 0.15
    new_with_cache = (
        (cached_tokens / 1_000_000) * new_cache_price
        + (uncached_tokens / 1_000_000) * new_input_price
        + (output_tokens / 1_000_000) * new_output_price
    )

    print(f"Actual cost (old pricing, no cache): ${old_total:.4f}")
    print(f"New pricing without caching: ${new_no_cache:.4f}")
    print(f"New pricing WITH 85% caching: ${new_with_cache:.4f}")
    print(
        f"\nSavings from caching: ${new_no_cache - new_with_cache:.4f} ({(new_no_cache - new_with_cache) / new_no_cache * 100:.1f}%)"
    )

    print("\n" + "=" * 60)
    print("IMPLEMENTATION RECOMMENDATION")
    print("-" * 60)
    print("To enable prompt caching in AnthropicPlayer:")
    print(
        '1. Use the beta client: client = anthropic.Anthropic(beta=["prompt-caching-2024-07-31"])'
    )
    print("2. Add cache_control to system message and early conversation history")
    print("3. Mark game state from previous days with cache_control")
    print("4. This would reduce input costs by ~82% on days 2-30")
