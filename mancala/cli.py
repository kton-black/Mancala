import argparse

from pandas import DataFrame
import matplotlib.pyplot as plt

from mancala.agents import ALL_AI_AGENTS, HumanAgent, RandomAgent, init_agent
from mancala.arena import play_arena
from mancala.mancala import MancalaEnv
from mancala.play import CLIGame

parser = argparse.ArgumentParser(description="Mancala playable on cli")
subparsers = parser.add_subparsers(dest="command")

play_parser = subparsers.add_parser("play")
play_parser.add_argument(
    "--player0",
    type=str,
    default="human",
    help="Player that makes the first move",
    choices=ALL_AI_AGENTS + ["human"],
)
play_parser.add_argument(
    "--player1",
    type=str,
    default="random",
    help="Player that makes move the next turn",
    choices=ALL_AI_AGENTS + ["human"],
)
play_parser.add_argument(
    "--depth",
    type=int,
    default=2,
    help="Depth for MiniMax and Negascout agents"
)

arena_parser = subparsers.add_parser("arena")
arena_parser.add_argument(
    "--num_games",
    type=int,
    default=100,
    help="How many times each pairs of agents should play together",
)
arena_parser.add_argument(
    "--only",
    type=str,
    default="",
    help="Explicitly select agents to compare from ",
)
arena_parser.add_argument(
    "--depth",
    type=int,
    default=2,
    help="Depth for MiniMax and Negascout agents"
)


def cli():
    args = parser.parse_args()
    if args.command == "play":
        agent0 = init_agent(args.player0, 0, args.depth)
        agent1 = init_agent(args.player1, 1, args.depth)
        env = MancalaEnv(player0=agent0, player1=agent1)
        game = CLIGame(env)
        game.play_cli()
    elif args.command == "arena":
        agents = ALL_AI_AGENTS
        if args.only:
            agents = args.only.split(",")
        wins, times = play_arena(agents, args.num_games, depth=args.depth)
        print("Wins (percent for p1 to win):")
        df_wins = DataFrame(wins)
        print(df_wins)
        df_wins.to_csv('outputs/wins.csv')
        df_wins.plot.bar()
        plt.tight_layout()
        plt.savefig('outputs/wins.png')
        print()
        print("Time:")
        df_times = DataFrame(times)
        print(df_times)
        df_times.to_csv('outputs/times.csv')
        df_times.plot.bar()
        plt.tight_layout()
        plt.savefig('outputs/times.png')

if __name__ == "__main__":
    cli()