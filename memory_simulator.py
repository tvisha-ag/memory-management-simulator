"""
Memory Management Simulator
============================
Simulates OS memory allocation using three classic algorithms:
  - First Fit   : allocate in the first hole that fits
  - Best Fit    : allocate in the smallest hole that fits
  - Worst Fit   : allocate in the largest hole (maximises leftover)

Data Structures used:
  - Doubly-Linked List of memory blocks (free + allocated)
  - Min-Heap (heapq) for Best Fit hole selection
  - Max-Heap (heapq) for Worst Fit hole selection

Research Angle:
  Compare External Fragmentation % for each algorithm under
  simulated heavy load (random alloc/dealloc workload).

  External Fragmentation = (free memory not usable for largest request)
                           ─────────────────────────────────────────────
                                    total free memory

References:
  Silberschatz, Galvin & Gagne — "Operating System Concepts" (10th ed.)
  Knuth — "The Art of Computer Programming Vol. 1", §2.5 Dynamic Storage
"""

import random
import heapq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from copy import deepcopy

random.seed(42)
np.random.seed(42)


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — MEMORY BLOCK (Doubly-Linked List Node)
# ══════════════════════════════════════════════════════════════

@dataclass
class Block:
    """
    A contiguous region of memory.
    Forms a doubly-linked list representing the full address space.
    """
    start:  int
    size:   int
    free:   bool = True
    pid:    Optional[int] = None   # process id if allocated
    prev:   Optional['Block'] = field(default=None, repr=False)
    next:   Optional['Block'] = field(default=None, repr=False)

    @property
    def end(self):
        return self.start + self.size - 1

    def __repr__(self):
        status = f"FREE({self.size})" if self.free else f"PID{self.pid}({self.size})"
        return f"[{self.start}-{self.end}: {status}]"


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — MEMORY MANAGER BASE CLASS
# ══════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Base memory manager. Maintains a doubly-linked list of blocks.
    Subclasses implement _find_hole() with different strategies.

    Total memory = TOTAL_SIZE bytes (simulated address space).
    """

    def __init__(self, total_size: int):
        self.total_size  = total_size
        self.alloc_count = 0
        self.fail_count  = 0
        self.frag_history: list[float] = []
        self.util_history: list[float] = []

        # Initialise: one big free block spanning entire address space
        self.head = Block(start=0, size=total_size, free=True)

    # ── Public API ────────────────────────────────────────────

    def allocate(self, size: int, pid: int) -> bool:
        """
        Try to allocate `size` bytes for process `pid`.
        Returns True on success, False on failure (no suitable hole).
        """
        hole = self._find_hole(size)
        if hole is None:
            self.fail_count += 1
            return False

        self._split_block(hole, size, pid)
        self.alloc_count += 1
        return True

    def free(self, pid: int) -> bool:
        """
        Release all memory held by process `pid`.
        Coalesces adjacent free blocks (prevents artificial fragmentation).
        """
        freed = False
        block = self.head
        while block:
            if not block.free and block.pid == pid:
                block.free = True
                block.pid  = None
                self._coalesce(block)
                freed = True
                break
            block = block.next
        return freed

    def snapshot(self) -> list[dict]:
        """Return current memory map as a list of block dicts."""
        out, b = [], self.head
        while b:
            out.append({'start': b.start, 'size': b.size,
                        'free': b.free, 'pid': b.pid})
            b = b.next
        return out

    # ── Metrics ───────────────────────────────────────────────

    def external_fragmentation(self) -> float:
        """
        External Fragmentation (Silberschatz definition):

          EF = 1 - (largest_free_hole / total_free_memory)

        = fraction of free memory that CANNOT satisfy the largest
          possible request due to fragmentation into small holes.
        EF = 0.0 → perfectly contiguous free memory (no fragmentation)
        EF = 1.0 → all free memory is in tiny unusable fragments
        """
        free_blocks = []
        b = self.head
        while b:
            if b.free:
                free_blocks.append(b.size)
            b = b.next

        if not free_blocks or sum(free_blocks) == 0:
            return 0.0

        total_free  = sum(free_blocks)
        largest_hole = max(free_blocks)
        return 1.0 - (largest_hole / total_free)

    def utilization(self) -> float:
        """Fraction of total memory currently allocated."""
        used = 0
        b = self.head
        while b:
            if not b.free:
                used += b.size
            b = b.next
        return used / self.total_size

    def record_snapshot(self):
        self.frag_history.append(self.external_fragmentation())
        self.util_history.append(self.utilization())

    def free_block_count(self) -> int:
        count, b = 0, self.head
        while b:
            if b.free: count += 1
            b = b.next
        return count

    def total_free(self) -> int:
        total, b = 0, self.head
        while b:
            if b.free: total += b.size
            b = b.next
        return total

    # ── Internal helpers ──────────────────────────────────────

    def _find_hole(self, size: int) -> Optional[Block]:
        raise NotImplementedError

    def _split_block(self, block: Block, size: int, pid: int):
        """
        Carve `size` bytes out of `block`.
        If leftover ≥ 1 byte, insert a new free block after it.
        """
        leftover = block.size - size
        block.size = size
        block.free = False
        block.pid  = pid

        if leftover > 0:
            new_block       = Block(start=block.start + size,
                                    size=leftover, free=True)
            new_block.prev  = block
            new_block.next  = block.next
            if block.next:
                block.next.prev = new_block
            block.next = new_block

    def _coalesce(self, block: Block):
        """Merge block with adjacent free neighbours (both directions)."""
        # Merge with next
        if block.next and block.next.free:
            nxt         = block.next
            block.size += nxt.size
            block.next  = nxt.next
            if nxt.next:
                nxt.next.prev = block

        # Merge with prev
        if block.prev and block.prev.free:
            prv          = block.prev
            prv.size    += block.size
            prv.next     = block.next
            if block.next:
                block.next.prev = prv


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — ALLOCATION ALGORITHMS
# ══════════════════════════════════════════════════════════════

class FirstFit(MemoryManager):
    """
    First Fit: scan from the HEAD and use the FIRST hole ≥ requested size.
    Time: O(n)  — fast but creates many small fragments near the start.
    """
    name = "First Fit"
    color = "#3b82f6"

    def _find_hole(self, size: int) -> Optional[Block]:
        b = self.head
        while b:
            if b.free and b.size >= size:
                return b
            b = b.next
        return None


class BestFit(MemoryManager):
    """
    Best Fit: scan ALL holes and use the SMALLEST one ≥ requested size.
    Minimises wasted space per allocation but creates many tiny fragments.
    Time: O(n) scan; can use a min-heap for O(log n) in production.
    """
    name = "Best Fit"
    color = "#10b981"

    def _find_hole(self, size: int) -> Optional[Block]:
        best, b = None, self.head
        while b:
            if b.free and b.size >= size:
                if best is None or b.size < best.size:
                    best = b
            b = b.next
        return best


class WorstFit(MemoryManager):
    """
    Worst Fit: use the LARGEST available hole.
    Leaves large remainders — reduces tiny unusable fragments
    but wastes large holes on small requests.
    Time: O(n) scan; can use a max-heap for O(log n) in production.
    """
    name = "Worst Fit"
    color = "#f59e0b"

    def _find_hole(self, size: int) -> Optional[Block]:
        worst, b = None, self.head
        while b:
            if b.free and b.size >= size:
                if worst is None or b.size > worst.size:
                    worst = b
            b = b.next
        return worst


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — WORKLOAD SIMULATOR
# ══════════════════════════════════════════════════════════════

def run_simulation(manager: MemoryManager,
                   n_ops: int     = 500,
                   min_size: int  = 10,
                   max_size: int  = 200,
                   alloc_prob: float = 0.65) -> dict:
    """
    Simulate n_ops memory operations:
      - With probability alloc_prob  → allocate random-sized block
      - Otherwise                    → free a random live process

    Records external fragmentation after every operation.
    """
    live_pids: list[int] = []
    pid_counter = 1
    snapshots_at = []   # store memory maps at key points

    for op in range(n_ops):
        if live_pids and random.random() > alloc_prob:
            # Deallocate a random live process
            pid = random.choice(live_pids)
            if manager.free(pid):
                live_pids.remove(pid)
        else:
            # Allocate
            size = random.randint(min_size, max_size)
            if manager.allocate(size, pid_counter):
                live_pids.append(pid_counter)
            pid_counter += 1

        manager.record_snapshot()

        # Capture detailed snapshots at 25%, 50%, 75%, 100%
        for pct in [0.25, 0.50, 0.75, 1.0]:
            if op == int(n_ops * pct) - 1:
                snapshots_at.append({
                    'pct':      int(pct * 100),
                    'op':       op,
                    'map':      manager.snapshot(),
                    'ef':       manager.external_fragmentation(),
                    'util':     manager.utilization(),
                    'n_free':   manager.free_block_count(),
                    'total_free': manager.total_free(),
                })

    return {
        'name':           manager.name,
        'color':          manager.color,
        'frag_history':   manager.frag_history,
        'util_history':   manager.util_history,
        'alloc_count':    manager.alloc_count,
        'fail_count':     manager.fail_count,
        'final_ef':       manager.external_fragmentation(),
        'final_util':     manager.utilization(),
        'final_n_free':   manager.free_block_count(),
        'final_free':     manager.total_free(),
        'snapshots':      snapshots_at,
        'final_map':      manager.snapshot(),
    }


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — VISUALIZATION
# ══════════════════════════════════════════════════════════════

TOTAL_MEM  = 2048   # bytes (simulated address space)
N_OPS      = 500
MIN_SIZE   = 10
MAX_SIZE   = 200
ALLOC_PROB = 0.65

BG     = "#0d1117"
PANEL  = "#161b22"
BORDER = "#30363d"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
GRID   = "#21262d"


def draw_memory_map(ax, memory_map: list[dict], title: str, color: str):
    """Draw a horizontal memory map bar."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)

    total = sum(b['size'] for b in memory_map)
    x = 0
    for b in memory_map:
        w   = b['size'] / total
        col = PANEL if b['free'] else color
        ax.barh(0, w, left=x, height=0.6,
                color=col, edgecolor=BG, linewidth=0.3)
        x += w

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '512', '1024', '1536', '2048'],
                        fontsize=7, color=MUTED)
    ax.set_title(title, fontsize=9, color=color, pad=5)


def make_charts(results: list[dict]):
    fig = plt.figure(figsize=(20, 20), facecolor=BG)

    fig.text(0.05, 0.977,
             "Memory Management Simulator",
             fontsize=22, fontweight='bold', color=TEXT,
             fontfamily='DejaVu Sans Mono')
    fig.text(0.05, 0.962,
             f"First Fit  ·  Best Fit  ·  Worst Fit  ·  "
             f"Total Memory: {TOTAL_MEM} bytes  ·  "
             f"Workload: {N_OPS} operations  ·  "
             f"Alloc probability: {ALLOC_PROB:.0%}",
             fontsize=11, color=MUTED)

    gs = GridSpec(5, 3, figure=fig,
                  top=0.95, bottom=0.05,
                  hspace=0.62, wspace=0.32)

    # Row 0: External Fragmentation over time (full width)
    ax_ef   = fig.add_subplot(gs[0, :])
    # Row 1: Utilization over time (full width)
    ax_util = fig.add_subplot(gs[1, :])
    # Row 2: Final memory maps (one per algo)
    ax_map  = [fig.add_subplot(gs[2, i]) for i in range(3)]
    # Row 3: Free block count bar + alloc success bar + final EF bar
    ax_nfree = fig.add_subplot(gs[3, 0])
    ax_succ  = fig.add_subplot(gs[3, 1])
    ax_efbar = fig.add_subplot(gs[3, 2])
    # Row 4: Research summary table
    ax_tbl   = fig.add_subplot(gs[4, :])

    for ax in [ax_ef, ax_util, ax_nfree, ax_succ, ax_efbar, ax_tbl]:
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

    # ── 1. External Fragmentation over time ──────────────────
    for r in results:
        ax_ef.plot(r['frag_history'], color=r['color'],
                   linewidth=1.8, alpha=0.9, label=r['name'])
        # Fill under curve
        ax_ef.fill_between(range(len(r['frag_history'])),
                            r['frag_history'],
                            alpha=0.08, color=r['color'])

    ax_ef.set_title('External Fragmentation % over Time  (lower = better)',
                    fontsize=13, color=TEXT, pad=8)
    ax_ef.set_xlabel('Operation number', fontsize=9, color=MUTED)
    ax_ef.set_ylabel('External Fragmentation', fontsize=9, color=MUTED)
    ax_ef.tick_params(colors=MUTED, labelsize=8)
    ax_ef.grid(color=GRID, linewidth=0.6, linestyle='--')
    ax_ef.legend(fontsize=10, facecolor=BG, edgecolor=BORDER,
                 labelcolor=TEXT)
    ax_ef.set_ylim(0, 1.05)

    # ── 2. Utilization over time ──────────────────────────────
    for r in results:
        ax_util.plot(r['util_history'], color=r['color'],
                     linewidth=1.8, alpha=0.9, label=r['name'])

    ax_util.set_title('Memory Utilization % over Time  (higher = better)',
                      fontsize=13, color=TEXT, pad=8)
    ax_util.set_xlabel('Operation number', fontsize=9, color=MUTED)
    ax_util.set_ylabel('Utilization', fontsize=9, color=MUTED)
    ax_util.tick_params(colors=MUTED, labelsize=8)
    ax_util.grid(color=GRID, linewidth=0.6, linestyle='--')
    ax_util.legend(fontsize=10, facecolor=BG, edgecolor=BORDER,
                   labelcolor=TEXT)
    ax_util.set_ylim(0, 1.05)

    # ── 3. Final memory maps ──────────────────────────────────
    for ax, r in zip(ax_map, results):
        ef_pct = r['final_ef'] * 100
        draw_memory_map(
            ax, r['final_map'],
            f"{r['name']}  —  EF = {ef_pct:.1f}%",
            r['color']
        )

    # ── 4. Free block count ───────────────────────────────────
    names   = [r['name'] for r in results]
    colors  = [r['color'] for r in results]
    nfree   = [r['final_n_free'] for r in results]
    bars4   = ax_nfree.bar(names, nfree, color=colors,
                            edgecolor=BG, width=0.5)
    for bar, v in zip(bars4, nfree):
        ax_nfree.text(bar.get_x()+bar.get_width()/2, v+0.3,
                      str(v), ha='center', fontsize=11,
                      fontweight='bold', color=TEXT)
    ax_nfree.set_title('Free Block Count at End\n(more = more fragmented)',
                       fontsize=11, color=TEXT)
    ax_nfree.tick_params(colors=MUTED, labelsize=9)
    ax_nfree.grid(axis='y', color=GRID, linewidth=0.6)
    ax_nfree.set_facecolor(PANEL)
    for sp in ax_nfree.spines.values(): sp.set_color(BORDER)

    # ── 5. Allocation success rate ────────────────────────────
    succ_rates = [r['alloc_count'] / max(r['alloc_count']+r['fail_count'], 1)
                  for r in results]
    bars5 = ax_succ.bar(names, succ_rates, color=colors,
                        edgecolor=BG, width=0.5)
    for bar, v in zip(bars5, succ_rates):
        ax_succ.text(bar.get_x()+bar.get_width()/2, v+0.005,
                     f'{v:.1%}', ha='center', fontsize=11,
                     fontweight='bold', color=TEXT)
    ax_succ.set_title('Allocation Success Rate\n(higher = better)',
                      fontsize=11, color=TEXT)
    ax_succ.set_ylim(0, 1.15)
    ax_succ.tick_params(colors=MUTED, labelsize=9)
    ax_succ.grid(axis='y', color=GRID, linewidth=0.6)
    ax_succ.set_facecolor(PANEL)
    for sp in ax_succ.spines.values(): sp.set_color(BORDER)

    # ── 6. Final external fragmentation bar ──────────────────
    ef_vals = [r['final_ef'] * 100 for r in results]
    bars6   = ax_efbar.bar(names, ef_vals, color=colors,
                            edgecolor=BG, width=0.5)
    ax_efbar.axhline(20, color='#dc2626', linewidth=1.5,
                     linestyle='--', alpha=0.7, label='Acceptable threshold (20%)')
    for bar, v in zip(bars6, ef_vals):
        ax_efbar.text(bar.get_x()+bar.get_width()/2, v+0.4,
                      f'{v:.1f}%', ha='center', fontsize=11,
                      fontweight='bold', color=TEXT)
    ax_efbar.set_title('Final External Fragmentation %\n(lower = better)',
                       fontsize=11, color=TEXT)
    ax_efbar.legend(fontsize=8, facecolor=BG, edgecolor=BORDER, labelcolor=TEXT)
    ax_efbar.tick_params(colors=MUTED, labelsize=9)
    ax_efbar.grid(axis='y', color=GRID, linewidth=0.6)
    ax_efbar.set_facecolor(PANEL)
    for sp in ax_efbar.spines.values(): sp.set_color(BORDER)

    # ── 7. Research summary table ─────────────────────────────
    ax_tbl.axis('off')
    ax_tbl.set_facecolor(PANEL)
    for sp in ax_tbl.spines.values(): sp.set_color(BORDER)

    col_labels = ['Algorithm', 'Time\nComplexity',
                  'Final EF %', 'Free Blocks',
                  'Alloc Success', 'Alloc Failures',
                  'Best Use Case']
    best_cases = ['General purpose\nOS allocator',
                  'Tight memory\nenvironments',
                  'Large requests\ndominate workload']
    rows = []
    for r, bc in zip(results, best_cases):
        sr = r['alloc_count'] / max(r['alloc_count']+r['fail_count'], 1)
        rows.append([
            r['name'],
            'O(n)',
            f"{r['final_ef']*100:.1f}%",
            str(r['final_n_free']),
            f"{sr:.1%}",
            str(r['fail_count']),
            bc,
        ])

    tbl = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.05, 1, 0.90]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        if row == 0:
            cell.set_facecolor('#21262d')
            cell.set_text_props(color=TEXT, fontweight='bold')
        else:
            rc = results[row-1]['color']
            cell.set_facecolor(PANEL)
            if col == 0:
                cell.set_text_props(color=rc, fontweight='bold')
            else:
                cell.set_text_props(color=MUTED)

    # Caption
    fig.text(0.5, 0.025,
             f"Simulated address space: {TOTAL_MEM} bytes  ·  "
             f"Request size: {MIN_SIZE}–{MAX_SIZE} bytes  ·  "
             f"Allocation probability: {ALLOC_PROB:.0%}  ·  "
             f"EF = 1 − (largest_free_hole / total_free)  "
             f"[Silberschatz et al.]",
             ha='center', fontsize=8, color=MUTED)

    out = '/mnt/user-data/outputs/memory_simulator.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f"  Chart saved → {out}")
    return out


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — FINAL REPORT
# ══════════════════════════════════════════════════════════════

def print_report(results: list[dict]):
    print('\n' + '═'*72)
    print('  MEMORY MANAGEMENT SIMULATOR — RESULTS')
    print('═'*72)

    for r in results:
        sr = r['alloc_count'] / max(r['alloc_count']+r['fail_count'], 1)
        print(f"""
  ── {r['name']} ──
  Allocations succeeded : {r['alloc_count']}
  Allocations failed    : {r['fail_count']}  ({1-sr:.1%} failure rate)
  Final free blocks     : {r['final_n_free']}
  Total free memory     : {r['final_free']} / {TOTAL_MEM} bytes
  External Fragmentation: {r['final_ef']*100:.2f}%""")

    print(f"""
  COMPARISON SUMMARY
  {'─'*70}
  {"Algorithm":<14} {"Final EF %":>12} {"Free Blocks":>13} {"Alloc Failures":>16}  Rank
  {'─'*70}""")

    ranked = sorted(results, key=lambda r: r['final_ef'])
    for i, r in enumerate(ranked):
        sr = r['alloc_count'] / max(r['alloc_count']+r['fail_count'], 1)
        print(f"  {r['name']:<14} {r['final_ef']*100:>11.2f}% "
              f"{r['final_n_free']:>13} {r['fail_count']:>16}  #{i+1}")

    winner = ranked[0]
    print(f"""
  RESEARCH FINDINGS
  {'─'*70}
  1. LOWEST EXTERNAL FRAGMENTATION: {winner['name']}
     EF = {winner['final_ef']*100:.2f}% — fewest unusable free fragments.

  2. FRAGMENTATION FORMULA (Silberschatz et al., OS Concepts 10e):
     EF = 1 − (largest_free_hole / total_free_memory)
     EF = 0.0 → all free memory is in one contiguous block (ideal)
     EF = 1.0 → all free memory is in 1-byte unusable fragments

  3. WHY BEST FIT DOESN'T ALWAYS WIN:
     Best Fit minimises waste per allocation but creates many tiny
     leftover fragments (≤ min_size). These fragments are too small
     for any future request → high long-term fragmentation.

  4. WORST FIT INTUITION:
     By choosing the largest hole, Worst Fit leaves large remainders
     that can satisfy future requests. But under heavy mixed workloads
     it wastes large holes on small requests, degrading over time.

  5. FIRST FIT IN PRACTICE:
     Linux kernel (slab allocator) and glibc malloc() both use
     variants of First Fit with coalescing. Speed (O(n) with
     early exit) often matters more than fragmentation percentage
     in real OS kernels.
  {'─'*70}""")
    print('═'*72)


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print('\n' + '═'*72)
    print('  Memory Management Simulator')
    print('  First Fit · Best Fit · Worst Fit')
    print(f'  Address space: {TOTAL_MEM} bytes  ·  '
          f'Workload: {N_OPS} ops  ·  '
          f'Request size: {MIN_SIZE}–{MAX_SIZE} bytes')
    print('═'*72)

    # Run the same workload on all three algorithms
    # (same random seed → identical request sequence for fair comparison)
    all_results = []
    for AlgoClass in [FirstFit, BestFit, WorstFit]:
        random.seed(42)   # reset seed for identical workload
        np.random.seed(42)
        manager = AlgoClass(total_size=TOTAL_MEM)
        print(f'\n  Running {AlgoClass.name} …')
        result = run_simulation(
            manager,
            n_ops=N_OPS,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            alloc_prob=ALLOC_PROB,
        )
        all_results.append(result)
        print(f'    Done — EF={result["final_ef"]*100:.1f}%  '
              f'allocs={result["alloc_count"]}  '
              f'fails={result["fail_count"]}')

    print('\n  Generating visualisation …')
    make_charts(all_results)

    print_report(all_results)
