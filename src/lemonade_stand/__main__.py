"""Command line interface for the lemonade stand benchmark."""

from __future__ import annotations

import argparse

from .simple_game import SimpleLemonadeGame


def main(argv: list[str] | None = None) -> None:
    """Run a simple simulation of the lemonade stand game.

    Parameters
    ----------
    argv:
        Optional list of command line arguments. If ``None`` the arguments
        are taken from ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Run a basic lemonade stand simulation"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to play",
    )
    parser.add_argument(
        "--price",
        type=float,
        help="Fixed price to charge each day (defaults to suggested starting price)",
    )
    args = parser.parse_args(argv)

    game = SimpleLemonadeGame(days=args.days)
    price = args.price if args.price is not None else game.suggested_starting_price

    print(f"Starting game for {args.days} days at ${price:.2f} per lemonade\n")
    for _ in range(args.days):
        result = game.play_turn(price=price)
        print(
            f"Day {result['day']:2d}: price=${result['price']:.2f}"
            f" customers={result['customers']:3d}"
            f" profit=${result['profit']:.2f}"
            f" cash=${result['cash']:.2f}"
        )
    print("Game over!")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
