# AI-project

🧠 Maze Safety Pathfinder
This is a web app that uses AI (Perceptron) to check if maze tiles are safe or unsafe, and then finds a safe path from a start point to an end point using the A* algorithm.


🚀 What It Does
Makes a random maze with grass, water, and obstacles

Uses a Perceptron model to check if each tile is safe

Finds a safe path from Start (S) to End (E) while avoiding unsafe and blocked tiles

Lets you edit the maze, change start/end points, and see the result


🖥️ How to Use
Open index.html in your browser

Choose maze size and tile types (grass %, water %, obstacle %)

Click Generate Maze

Set Start and End points

Click Find Path to see the safe route

You can also use Edit Mode to change tiles manually


📁 Files
File	Use
index.html	The main webpage
script.js	All the AI logic and maze functions
styles.css	Page styles
students.txt	Student names
Data.xlsx	(Optional) data file for AI training

👨‍🎓 Made By
Osama Ishtaiwi – 12112273

Jad Kawa – 12112971


💡 What’s Inside
A simple Perceptron model that learns from sample data

A* search algorithm to find the best path

Smart tile system using:

Type (grass = 0, water = 1, obstacle = 2)

Elevation (0–10)

Distance to obstacle


✅ Future Ideas
Use real training data from the Excel file

Add new terrain types and movement costs

Improve AI with more advanced models

