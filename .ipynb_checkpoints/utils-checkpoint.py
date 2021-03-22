{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_velocity(action):\n",
    "    if action == 1:\n",
    "        return 12, 15\n",
    "    if action == 2:\n",
    "        return 10, 10\n",
    "    if action == 3:\n",
    "        return 0, 0\n",
    "    if action == 4:\n",
    "        return 12, 15\n",
    "    if action == 5:\n",
    "        return 0, 0\n",
    "\n",
    "def get_angle(action):\n",
    "    if action == 1:\n",
    "        return 0\n",
    "    if action == 2:\n",
    "        return 15\n",
    "    if action == 3:\n",
    "        return 0\n",
    "    if action == 4:\n",
    "        return 0\n",
    "    if action == 5:\n",
    "        return 0\n",
    "\n",
    "def get_position(action):\n",
    "    if action == 1:\n",
    "        return 10, 100\n",
    "    if action == 2:\n",
    "        return 50, 0\n",
    "    if action == 3:\n",
    "        return 50, 0\n",
    "    if action == 4:\n",
    "        return 50, 20\n",
    "    if action == 5:\n",
    "        return 2, 0\n",
    "\n",
    "def get_landing_zone():\n",
    "    return 50, 0\n",
    "\n",
    "def get_fuel(action):\n",
    "    if action == 1:\n",
    "        return 10\n",
    "    if action == 2:\n",
    "        return 20\n",
    "    if action == 3:\n",
    "        return 5\n",
    "    if action == 4:\n",
    "        return 0\n",
    "    if action == 5:\n",
    "        return 10\n",
    "\n",
    "def tests(LunarLander, test_number):\n",
    "    ll = LunarLander()\n",
    "    ll.env_start()\n",
    "    reward, obs, term = ll.env_step(test_number)\n",
    "    print(\"Reward: {}, Terminal: {}\".format(reward, term))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
