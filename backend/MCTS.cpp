#include <bits/stdc++.h>
using namespace std;
fstream f("game.txt");

struct game
{
    unsigned short board[6][7];
    short turn;

    void read()
    {
        f >> turn;
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 7; ++j)
                f >> board[i][j];
    }

    short play(short move)
    {
        int winner = 0;
        for (int i = 5; i >= 0; --i)
        {
            if (!board[i][move])
            {
                board[i][move] = turn;
                turn = 3 - turn;
                return overCheck(i, move);
            }
        }
        return 0;
    }
    short score(int player)
    {
        if (player == turn)
            return 1;
        return -1;
    }
    short overCheck(int x, int y)
    {
        int cnt;
        int val = board[x][y];
        // vertical
        cnt = 0;
        for (int i = x - 1; i >= 0; --i)
        {
            if (board[i][y] == val)
                ++cnt;
            else
                break;
        }
        for (int i = x + 1; i < 6; ++i)
        {
            if (board[i][y] == val)
                ++cnt;
            else
                break;
        }
        if (cnt >= 3)
            return val;

        // horisontal
        cnt = 0;
        for (int j = y - 1; j >= 0; --j)
        {
            if (board[x][j] == val)
                ++cnt;
            else
                break;
        }
        for (int j = y + 1; j < 7; ++j)
        {
            if (board[x][j] == val)
                ++cnt;
            else
                break;
        }
        if (cnt >= 3)
            return val;

        // diagonals
        cnt = 0;
        for (int i = x - 1, j = y - 1; i >= 0 && j >= 0; --i, --j)
        {
            if (board[i][j] == val)
                ++cnt;
            else
                break;
        }
        for (int i = x + 1, j = y + 1; i < 6 && j < 7; ++i, ++j)
        {
            if (board[i][j] == val)
                ++cnt;
            else
                break;
        }
        if (cnt >= 3)
            return val;

        cnt = 0;
        for (int i = x - 1, j = y + 1; i >= 0 && j < 7; --i, ++j)
        {
            if (board[i][j] == val)
                ++cnt;
            else
                break;
        }
        for (int i = x + 1, j = y - 1; i < 6 && j >= 0; ++i, --j)
        {
            if (board[i][j] == val)
                ++cnt;
            else
                break;
        }
        if (cnt >= 3)
            return val;

        // else, no winner
        return 0;
    }
};

tuple<int, int> rollout(game &init, default_random_engine &engine)
{
    int v[] = {0, 1, 2, 3, 4, 5, 6};
    const int size = sizeof(v) / sizeof(v[0]);
    game myGame = init;
    int rootMove = -1;
    for (;;)
    {

        shuffle(v, v + size, engine);

        short winner = 0;
        for (int i = 0; i < 7; ++i) // p1
        {
            if (!myGame.board[0][v[i]])
            {
                winner = myGame.play(v[i]);
                if (rootMove == -1)
                    rootMove = v[i];
                // cout << "p1 played " << v[i] << '\n';
                break;
            }
        }
        if (winner)
            return make_tuple(init.score(winner), rootMove);

        bool gameOver = true;
        for (int i = 0; i < 7; ++i)
        {
            if (!myGame.board[0][i])
                gameOver = false;
        }

        if (gameOver)
            return make_tuple(0, rootMove);

        shuffle(begin(v), end(v), engine);
        for (int i = 0; i < 7; ++i) // p2
        {
            if (!myGame.board[0][v[i]])
            {
                winner = myGame.play(v[i]);
                // cout << "p2 played " << v[i] << '\n';

                break;
            }
        }
        if (winner)
            return make_tuple(init.score(winner), rootMove);

        for (int i = 0; i < 7; ++i)
        {
            if (!myGame.board[0][i])
                gameOver = false;
        }

        if (gameOver)
            return make_tuple(0, rootMove);
    }
    return {0, 0};
}

int main()
{
    srand(time(0));
    random_device rd;
    default_random_engine engine(rd());
    game myGame;
    myGame.read();

    int score[7];
    int visits[7];
    memset(score, 0, sizeof(score));
    memset(visits, 0, sizeof(visits));
    int iterations = 100000;
    for (; --iterations;)
    {
        int result, move;

        tie(result, move) = rollout(myGame, engine);
        score[move] += result;
        visits[move]++;
    }

    int maxi = -1000000, rez = -1;
    for (int i = 0; i < 7; ++i)
        if (score[i] > maxi && !myGame.board[0][i])
            maxi = score[i], rez = i;


    // ofstream gtest("game.txt");
    // gtest << myGame.turn << '\n';
    // myGame.play(rez);
    // for (int i = 0; i < 6; ++i)
    // {
    //     for (int j = 0; j < 7; ++j)
    //         gtest << myGame.board[i][j] << " ";
    //     gtest << '\n';
    // }
    cout << rez << '\n';
}