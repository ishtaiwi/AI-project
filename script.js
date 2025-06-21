class Perceptron {
    constructor() {
        this.weights = [Math.random(), Math.random(), Math.random()];
        this.bias = Math.random();
        this.learningRate = 0.1;
    }
    
    activate(sum) {
        return sum > 0 ? 1 : 0;
    }
    
    predict(features) {
        let sum = this.bias;
        for (let i = 0; i < this.weights.length; i++) {
            sum += features[i] * this.weights[i];
        }
        return this.activate(sum);
    }
    
    train(features, label) {
        const prediction = this.predict(features);
        const error = label - prediction;
        
        for (let i = 0; i < this.weights.length; i++) {
            this.weights[i] += this.learningRate * error * features[i];
        }
        
        this.bias += this.learningRate * error;
        return error;
    }
    
    trainBatch(data, epochs) {
        for (let e = 0; e < epochs; e++) {
            let totalError = 0;
            for (const example of data) {
                const error = this.train(example.features, example.label);
                totalError += Math.abs(error);
            }
            if (totalError === 0) break;
        }
    }
}

class Maze {
    constructor(size) {
        this.size = size;
        this.grid = [];
        this.start = null;
        this.end = null;
        this.perceptron = new Perceptron();
        this.path = [];
        this.trainingData = this.getDefaultTrainingData();
        this.editMode = false;
        
        if (this.trainingData) {
            this.perceptron.trainBatch(this.trainingData, 200);
        }
    }
    
    getDefaultTrainingData() {
        return [
            {"features": [0,1,10], "label": 1},
            {"features": [1,9,8], "label": 0},
            {"features": [0,1,4], "label": 1},
            {"features": [0,9,0], "label": 1},
            {"features": [0,3,2], "label": 1},
            {"features": [1,7,9], "label": 0},
            {"features": [0,6,7], "label": 1},
            {"features": [0,8,10], "label": 1},
            {"features": [0,7,5], "label": 1},
            {"features": [1,4,7], "label": 0},
            {"features": [0,1,8], "label": 1},
            {"features": [0,4,3], "label": 1},
            {"features": [0,7,0], "label": 1},
            {"features": [0,9,0], "label": 1},
            {"features": [1,8,9], "label": 0},
            {"features": [0,8,3], "label": 1},
            {"features": [1,0,6], "label": 0},
            {"features": [1,8,1], "label": 0},
            {"features": [1,6,2], "label": 0},
            {"features": [0,8,0], "label": 1},
            {"features": [1,7,4], "label": 0},
            {"features": [0,0,0], "label": 0},
            {"features": [1,7,7], "label": 0},
            {"features": [1,7,0], "label": 0},
            {"features": [1,10,10], "label": 0},
            {"features": [1,2,0], "label": 0},
            {"features": [1,0,1], "label": 0},
            {"features": [1,7,1], "label": 0},
            {"features": [1,2,5], "label": 0},
            {"features": [1,2,6], "label": 0},
            {"features": [0,0,4], "label": 1},
            {"features": [0,10,0], "label": 1},
            {"features": [1,4,0], "label": 0},
            {"features": [1,9,2], "label": 0},
            {"features": [1,6,1], "label": 0},
            {"features": [0,9,4], "label": 1},
            {"features": [1,8,9], "label": 0},
            {"features": [0,6,5], "label": 1},
            {"features": [0,8,6], "label": 1},
            {"features": [0,7,3], "label": 1},
            {"features": [0,1,6], "label": 1},
            {"features": [0,0,10], "label": 1},
            {"features": [1,6,7], "label": 0},
            {"features": [1,6,10], "label": 0},
            {"features": [1,7,0], "label": 0},
            {"features": [1,4,5], "label": 0},
            {"features": [1,2,7], "label": 0},
            {"features": [0,7,4], "label": 1},
            {"features": [1,5,3], "label": 0},
            {"features": [1,10,1], "label": 0},
            {"features": [0,2,5], "label": 1},
            {"features": [1,0,5], "label": 0},
            {"features": [0,2,10], "label": 1},
            {"features": [1,4,0], "label": 0},
            {"features": [0,2,8], "label": 1},
            {"features": [1,0,10], "label": 0},
            {"features": [1,4,5], "label": 0},
            {"features": [0,9,2], "label": 1},
            {"features": [0,6,3], "label": 1},
            {"features": [0,6,10], "label": 1},
            {"features": [0,10,3], "label": 1},
            {"features": [0,8,2], "label": 1},
            {"features": [0,9,9], "label": 1},
            {"features": [0,9,2], "label": 1},
            {"features": [0,2,2], "label": 0},
            {"features": [1,6,3], "label": 0},
            {"features": [1,0,6], "label": 0},
            {"features": [0,3,3], "label": 1},
            {"features": [1,3,8], "label": 0},
            {"features": [1,4,0], "label": 0},
            {"features": [1,6,7], "label": 0},
            {"features": [1,6,6], "label": 0},
            {"features": [0,10,1], "label": 1},
            {"features": [1,3,7], "label": 0},
            {"features": [0,6,0], "label": 1},
            {"features": [1,10,10], "label": 0},
            {"features": [1,2,8], "label": 0},
            {"features": [1,5,8], "label": 0},
            {"features": [0,1,1], "label": 0},
            {"features": [1,9,6], "label": 0},
            {"features": [0,8,9], "label": 1},
            {"features": [1,4,2], "label": 0},
            {"features": [0,5,6], "label": 1},
            {"features": [1,3,9], "label": 0},
            {"features": [0,10,8], "label": 1},
            {"features": [0,9,3], "label": 1},
            {"features": [1,6,0], "label": 0},
            {"features": [0,8,1], "label": 1},
            {"features": [1,6,0], "label": 0},
            {"features": [1,0,4], "label": 0},
            {"features": [1,0,4], "label": 0},
            {"features": [1,8,10], "label": 0},
            {"features": [1,10,6], "label": 0},
            {"features": [1,8,8], "label": 0},
            {"features": [1,3,8], "label": 0},
            {"features": [1,8,2], "label": 0},
            {"features": [1,2,2], "label": 0},
            {"features": [1,6,2], "label": 0},
            {"features": [1,5,3], "label": 0},
            {"features": [0,7,7], "label": 1}
        ];
    }
    
    initialize(grassProb, waterProb, obstacleProb) {
        this.grid = [];
        this.path = [];
        
        const total = grassProb + waterProb + obstacleProb;
        const grassNorm = grassProb / total;
        const waterNorm = waterProb / total;
        
        for (let y = 0; y < this.size; y++) {
            const row = [];
            for (let x = 0; x < this.size; x++) {
                const rand = Math.random();
                let type, elevation;
                
                if (rand < grassNorm) {
                    type = 0;
                    elevation = Math.floor(Math.random() * 11);
                } else if (rand < grassNorm + waterNorm) {
                    type = 1;
                    elevation = Math.floor(Math.random() * 11);
                } else {
                    type = 2;
                    elevation = 0;
                }
                
                row.push({
    type: type,
    elevation: elevation,
    distance: 0,
    unsafe: undefined 
});
            }
            this.grid.push(row);
        }
        
        this.calculateObstacleDistances();
        this.setStartEndPoints();
        this.classifyTiles();
        
        return this.grid;
    }
    
    calculateObstacleDistances() {
        for (let y = 0; y < this.size; y++) {
            for (let x = 0; x < this.size; x++) {
                if (this.grid[y][x].type === 2) {
                    this.grid[y][x].distance = 0;
                    continue;
                }
                
                let minDistance = Infinity;
                
                for (let oy = 0; oy < this.size; oy++) {
                    for (let ox = 0; ox < this.size; ox++) {
                        if (this.grid[oy][ox].type === 2) {
                            const dist = Math.abs(x - ox) + Math.abs(y - oy);
                            if (dist < minDistance) {
                                minDistance = dist;
                            }
                        }
                    }
                }
                
                this.grid[y][x].distance = minDistance;
            }
        }
    }
    
    setStartEndPoints() {
        let startX, startY;
        do {
            startX = Math.floor(Math.random() * this.size);
            startY = Math.floor(Math.random() * this.size);
        } while (this.grid[startY][startX].type === 2);
        
        this.start = {x: startX, y: startY};
        
        let endX, endY;
        do {
            endX = Math.floor(Math.random() * this.size);
            endY = Math.floor(Math.random() * this.size);
        } while (
            this.grid[endY][endX].type === 2 || 
            (endX === startX && endY === startY)
        );
        
        this.end = {x: endX, y: endY};
    }
    
   classifyTiles() {
    for (let y = 0; y < this.size; y++) {
        for (let x = 0; x < this.size; x++) {
            const tile = this.grid[y][x];
            
            
            if (tile.type === 2) {
                tile.safe = false;
                continue;
            }
            
            if ((x === this.start.x && y === this.start.y) || 
                (x === this.end.x && y === this.end.y)) {
                tile.safe = true;
                continue;
            }
            
            const features = [tile.type, tile.elevation, tile.distance];
            tile.safe = this.perceptron.predict(features) === 1;
        }
    }
}
    
 findPath() {
    for (let y = 0; y < this.size; y++) {
        for (let x = 0; x < this.size; x++) {
            const tile = this.grid[y][x];
            
            if (tile.type === 2) {
                tile.unsafe = true;
                continue;
            }
            
            if ((x === this.start.x && y === this.start.y) || 
                (x === this.end.x && y === this.end.y)) {
                tile.unsafe = false;
                continue;
            }
            
            const features = [tile.type, tile.elevation, tile.distance];
            tile.unsafe = this.perceptron.predict(features) === 0;
        }
    }
    
    this.path = [];
    const openSet = new PriorityQueue();
    const cameFrom = new Map();
    
    const gScore = new Array(this.size).fill().map(() => new Array(this.size).fill(Infinity));
    gScore[this.start.y][this.start.x] = 0;
    
    const fScore = new Array(this.size).fill().map(() => new Array(this.size).fill(Infinity));
    fScore[this.start.y][this.start.x] = this.heuristic(this.start, this.end);
    
    openSet.enqueue(this.start, fScore[this.start.y][this.start.x]);
    
    while (!openSet.isEmpty()) {
        const current = openSet.dequeue();
        
        if (current.x === this.end.x && current.y === this.end.y) {
            this.path = this.reconstructPath(cameFrom, current);
            return "Path found!";
        }
        
        for (const neighbor of this.getNeighbors(current)) {
            const tile = this.grid[neighbor.y][neighbor.x];
            
            if (tile.type === 2 || tile.unsafe) {
                continue;
            }
            
            const tentativeGScore = gScore[current.y][current.x] + 1;
            
            if (tentativeGScore < gScore[neighbor.y][neighbor.x]) {
                cameFrom.set(`${neighbor.x},${neighbor.y}`, current);
                gScore[neighbor.y][neighbor.x] = tentativeGScore;
                fScore[neighbor.y][neighbor.x] = tentativeGScore + this.heuristic(neighbor, this.end);
                
                if (!openSet.contains(neighbor)) {
                    openSet.enqueue(neighbor, fScore[neighbor.y][neighbor.x]);
                }
            }
        }
    }
    
    return "No safe path found!";
}
    
    heuristic(a, b) {
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
    }
    
    getNeighbors(pos) {
        const neighbors = [];
        const directions = [
            {x: 0, y: -1},
            {x: 1, y: 0},
            {x: 0, y: 1},
            {x: -1, y: 0}
        ];
        
        for (const dir of directions) {
            const x = pos.x + dir.x;
            const y = pos.y + dir.y;
            
            if (x >= 0 && x < this.size && y >= 0 && y < this.size) {
                neighbors.push({x, y});
            }
        }
        
        return neighbors;
    }
    
    reconstructPath(cameFrom, current) {
        const path = [current];
        
        while (cameFrom.has(`${current.x},${current.y}`)) {
            current = cameFrom.get(`${current.x},${current.y}`);
            path.unshift(current);
        }
        
        return path;
    }
    
    enableEditMode() {
        this.editMode = true;
        document.getElementById('editMode').classList.add('edit-mode-active');
        this.render();
    }
    
    disableEditMode() {
        this.editMode = false;
        document.getElementById('editMode').classList.remove('edit-mode-active');
        this.render();
    }
    
    setStartPoint(x, y) {
        if (this.grid[y][x].type !== 2) { 
            this.start = {x, y};
            this.classifyTiles();
            this.render();
        }
    }
    
    setEndPoint(x, y) {
        if (this.grid[y][x].type !== 2) { 
            this.end = {x, y};
            this.classifyTiles();
            this.render();
        }
    }
    
   editCell(x, y) {
    const cellEditor = document.querySelector('.cell-editor');
    const tile = this.grid[y][x];
    
    const cellElement = document.querySelector(`table tr:nth-child(${y+1}) td:nth-child(${x+1})`);
    const rect = cellElement.getBoundingClientRect();
    cellEditor.style.left = `${rect.right + 10}px`;
    cellEditor.style.top = `${rect.top}px`;
    cellEditor.style.transform = 'none';
    cellEditor.style.display = 'block';
    
    const editorButtons = document.querySelectorAll('.cell-editor-buttons button');
    editorButtons.forEach(button => {
        button.onclick = () => {
            if (button.classList.contains('editor-grass')) {
                tile.type = 0;
                tile.elevation = Math.floor(Math.random() * 11);
                tile.unsafe = undefined; 
            } else if (button.classList.contains('editor-water')) {
                tile.type = 1;
                tile.elevation = Math.floor(Math.random() * 11);
                tile.unsafe = undefined; 
            } else if (button.classList.contains('editor-obstacle')) {
                tile.type = 2;
                tile.elevation = 0;
                tile.unsafe = true; 
            } else if (button.classList.contains('editor-elevation')) {
                const newElevation = prompt("Enter elevation (0-10):", tile.elevation);
                if (newElevation !== null) {
                    tile.elevation = Math.min(10, Math.max(0, parseInt(newElevation) )|| 0);
                    tile.unsafe = undefined; 
                }
            }
            
            
            this.calculateObstacleDistances();
            this.render();
            cellEditor.style.display = 'none';
        };
    });
    
   
    document.querySelector('.editor-cancel').onclick = () => {
        cellEditor.style.display = 'none';
    };
}
    
  render() {
    const container = document.getElementById('maze');
    container.innerHTML = '';
    
    const table = document.createElement('table');
    
    for (let y = 0; y < this.size; y++) {
        const row = document.createElement('tr');
        
        for (let x = 0; x < this.size; x++) {
            const cell = document.createElement('td');
            const tile = this.grid[y][x];
            
            
            if (this.editMode) {
                cell.style.cursor = 'pointer';
                cell.style.boxShadow = '0 0 0 2px var(--accent)';
            }
            
            if (x === this.start.x && y === this.start.y) {
                cell.className = 'start';
                cell.textContent = 'S';
            } else if (x === this.end.x && y === this.end.y) {
                cell.className = 'end';
                cell.textContent = 'E';
            } else if (this.path.some(p => p.x === x && p.y === y)) {
                cell.className = 'path';
                cell.textContent = '•';
            } else {
                switch (tile.type) {
                    case 0: cell.className = 'grass'; break;
                    case 1: cell.className = 'water'; break;
                    case 2: cell.className = 'obstacle'; break;
                }
                
                
                if (tile.unsafe === true && tile.type !== 2) {
                    cell.classList.add('unsafe');
                }
                
                const infoDiv = document.createElement('div');
                infoDiv.className = 'tile-info';
                infoDiv.textContent = `E:${tile.elevation} D:${tile.distance}`;
                cell.appendChild(infoDiv);
            }
            
            
            if (this.editMode) {
                cell.addEventListener('click', () => {
                    this.editCell(x, y);
                });
            }
            
            row.appendChild(cell);
        }
        
        table.appendChild(row);
    }
    
    container.appendChild(table);
}
}

class PriorityQueue {
    constructor() {
        this.elements = [];
    }
    
    enqueue(element, priority) {
        this.elements.push({element, priority});
        this.elements.sort((a, b) => a.priority - b.priority);
    }
    
    dequeue() {
        return this.elements.shift().element;
    }
    
    isEmpty() {
        return this.elements.length === 0;
    }
    
    contains(element) {
        return this.elements.some(e => e.element.x === element.x && e.element.y === element.y);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const mazeSize = document.getElementById('mazeSize');
    const grassProb = document.getElementById('grassProb');
    const waterProb = document.getElementById('waterProb');
    const obstacleProb = document.getElementById('obstacleProb');
    const generateBtn = document.getElementById('generateMaze');
    const findPathBtn = document.getElementById('findPath');
    const setStartBtn = document.getElementById('setStartPoint');
    const setEndBtn = document.getElementById('setEndPoint');
    const editModeBtn = document.getElementById('editMode');
    const outputDiv = document.getElementById('output');
    
    let maze = null;
    let settingStart = false;
    let settingEnd = false;

    generateBtn.addEventListener('click', () => {
        const size = parseInt(mazeSize.value);
        maze = new Maze(size);
        maze.initialize(
            parseInt(grassProb.value),
            parseInt(waterProb.value),
            parseInt(obstacleProb.value)
        );
        maze.render();
        outputDiv.textContent = 'New maze generated with trained perceptron!';
    });
    
   findPathBtn.addEventListener('click', () => {
    if (!maze) {
        outputDiv.textContent = 'Please generate a maze first!';
        return;
    }
    
    maze.calculateObstacleDistances();
    
    const result = maze.findPath();
    maze.render();
    
    if (result === "Path found!") {
        outputDiv.textContent = "Found a safe path!";
    } else {
        outputDiv.textContent = "No safe path could be found. Try adjusting the maze or safety parameters.";
    }
});
    
    setStartBtn.addEventListener('click', () => {
        settingStart = true;
        settingEnd = false;
        outputDiv.textContent = 'Click on a cell to set as start point';
    });
    
    setEndBtn.addEventListener('click', () => {
        settingEnd = true;
        settingStart = false;
        outputDiv.textContent = 'Click on a cell to set as end point';
    });
    
    editModeBtn.addEventListener('click', () => {
    if (!maze) {
        outputDiv.textContent = 'Please generate a maze first!';
        return;
    }
    
    if (maze.editMode) {
        maze.disableEditMode();
        outputDiv.textContent = 'Edit mode disabled';
    } else {
        maze.enableEditMode();
        outputDiv.textContent = 'Edit mode enabled - click cells to edit';
        maze.path = [];
    }
    maze.render();
});
    
    document.getElementById('maze').addEventListener('click', (e) => {
        if (!maze || (!settingStart && !settingEnd && !maze.editMode)) return;
        
        const cell = e.target.closest('td');
        if (!cell) return;
        
        const row = cell.parentElement;
        const x = Array.from(row.children).indexOf(cell);
        const y = Array.from(row.parentElement.children).indexOf(row);
        
        if (settingStart) {
            maze.setStartPoint(x, y);
            outputDiv.textContent = `Start point set to (${x}, ${y})`;
            settingStart = false;
        } else if (settingEnd) {
            maze.setEndPoint(x, y);
            outputDiv.textContent = `End point set to (${x}, ${y})`;
            settingEnd = false;
        }
    });
    
    maze = new Maze(parseInt(mazeSize.value));
    maze.initialize(
        parseInt(grassProb.value),
        parseInt(waterProb.value),
        parseInt(obstacleProb.value)
    );
    maze.render();
});