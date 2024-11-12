from nim import train, play, plot_eval
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner et jouer au jeu Nim avec DQN.")
    parser.add_argument("--train_games", type=int, default=100000, help="Nombre de jeux pour l'entraînement")
    args = parser.parse_args()

    # Entraînement
    ai, eval = train(args.train_games)
    plot_eval(eval)

    # Jouer contre l'IA
    play(ai)
