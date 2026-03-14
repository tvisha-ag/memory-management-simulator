# OS Memory Management Simulation & Fragmentation Analysis
> **Empirical validation of memory allocation heuristics and the "Best Fit Paradox" in Operating Systems.**

## 🌐 [Live Research Demo](https://tvisha-ag.github.io/memory-management-simulator/)
*(Note: Replace the URL above with your actual GitHub Pages link after enabling it in Settings > Pages)*

## 1. Executive Summary
This project is an experimental research tool designed to analyze the efficiency of **Contiguous Memory Allocation** strategies. By simulating 500+ operations on a 2048-byte address space, the study provides empirical evidence for the **"Best Fit Paradox"**—proving that the Best Fit algorithm, while theoretically precise, produces the highest rate of external fragmentation in high-load, mixed-workload environments.



## 2. Experimental Results & Findings
The simulation utilized a heavy-load benchmark of 500 mixed `malloc` and `free` operations. The results significantly contradict the intuitive assumption that "Best Fit" is superior for memory conservation.

### **Benchmark Comparison Data**
| Algorithm | Final Ext. Fragmentation | Free Blocks | Allocation Failures | Research Rank |
| :--- | :--- | :--- | :--- | :--- |
| **Worst Fit** | **66.67%** | 16 | 122 | **#1 (Optimal)** |
| **First Fit** | 72.71% | 11 | 132 | #2 |
| **Best Fit** | **82.60%** | 16 | 124 | #3 (Least Efficient) |

### **The "Tiny Sliver" Discovery**
The data demonstrates that **Best Fit** performs worst because it always chooses the tightest-fitting hole. This leaves behind "slivers" (fragments too small for any future request), eventually leading to an 82.6% fragmentation rate. Conversely, **Worst Fit** leaves behind larger contiguous holes that remain usable for subsequent processes, proving more resilient under stress.

## 3. Technical Implementation
The simulator uses a custom **Doubly-Linked List** architecture to model the heap accurately.

* **Logic Engine (`memory_simulator.py`):** A Python-based research tool using a `Block` class with `prev` and `next` pointers. It implements bidirectional **block coalescing**—when a block is freed, it automatically merges with adjacent free neighbors in $O(1)$ time to maximize contiguous space.
* **Web Dashboard (`index.html`):** A JavaScript port of the Python engine, providing a real-time visual representation of memory "holes" and process blocks. 
* **Complexity Analysis:** * **First Fit:** $O(n)$ search, $O(1)$ allocation.
    * **Best/Worst Fit:** $O(n)$ search to find the optimal/extreme hole.
    * **Coalescing:** $O(1)$ using doubly-linked pointers.



## 4. Key Features
- **Deterministic Workload Generation:** Adjustable total memory, request size, and allocation probability.
- **Real-Time Analytics:** Continuous calculation of External Fragmentation using the formula:
  $$EF = \frac{\text{Total Free Memory} - \text{Largest Contiguous Block}}{\text{Total Free Memory}} \times 100$$
- **Visual Analytics:** (Python version) Generates 8-panel plots showing fragmentation trends and memory pressure over time.

## 5. Setup & Usage
1.  **For Web:** Simply open the `index.html` in your browser or visit the Live Demo link.
2.  **For Python Analysis:**
    ```bash
    python memory_simulator.py
    ```
    This will run the 500-operation benchmark and output the `memory_simulator.png` analytics chart.

## 6. Academic Context
This research project serves as a practical implementation of concepts discussed in **Silberschatz’s Operating System Concepts (Chapter 9)**, providing a data-driven perspective on memory management trade-offs and heap stability.

---
*Created as part of a Research Portfolio for the IIT Hyderabad SURE Internship application.*
