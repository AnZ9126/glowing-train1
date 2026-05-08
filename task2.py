"""
Job Shop Scheduling Problem (10x10) — Optimized Solver
Algorithm: Iterated Local Search + Critical Path Neighborhood (Nowicki & Smutnicki N5)
           + Tabu Search + Multiple Constructive Heuristic Initialization

The N5 neighborhood swaps adjacent operations on the same machine that lie on the
critical path — the only moves that can possibly reduce the makespan.
"""

import random
import time
from collections import defaultdict

# ===================== 10 jobs, each with 10 operations =====================
# Format: each job = [(machine_id, processing_time), ...] in strict order
JOBS = [
    [(7,62), (8,24), (5,25), (3,84), (4,47), (6,38), (2,82), (0,93), (9,24), (1,66)],
    [(5,47), (2,97), (8,92), (9,22), (1,93), (4,29), (7,56), (3,80), (0,78), (6,67)],
    [(1,45), (7,46), (6,22), (2,26), (9,38), (0,69), (4,40), (3,33), (8,75), (5,96)],
    [(4,85), (8,76), (5,68), (9,88), (3,36), (6,75), (2,56), (1,35), (0,77), (7,85)],
    [(8,60), (9,20), (7,25), (3,63), (4,81), (0,52), (1,30), (5,98), (6,54), (2,86)],
    [(3,87), (9,73), (5,51), (2,95), (4,65), (1,86), (6,22), (8,58), (0,80), (7,65)],
    [(5,81), (2,53), (7,57), (6,71), (9,81), (0,43), (4,26), (8,54), (3,58), (1,69)],
    [(4,20), (6,86), (5,21), (8,79), (9,62), (2,34), (0,27), (1,81), (7,30), (3,46)],
    [(9,68), (6,66), (5,98), (8,86), (7,66), (0,56), (3,82), (1,95), (4,47), (2,78)],
    [(0,30), (3,50), (7,34), (2,58), (1,77), (5,34), (8,84), (4,40), (9,46), (6,44)]
]

NUM_JOBS = 10
NUM_MACHINES = 10
OPS_PER_JOB = 10
TOTAL_OPS = 100

# Pre-extract machine and time arrays for fast lookup
JOB_M = [[op[0] for op in job] for job in JOBS]
JOB_T = [[op[1] for op in job] for job in JOBS]


class JobShopSolver:
    def __init__(self, time_limit=30):
        self.time_limit = time_limit

    # ------------------------------------------------------------------
    # Decode: permutation → schedule → makespan + metadata
    # ------------------------------------------------------------------
    def decode(self, code):
        """Semi-active schedule builder. Returns makespan and schedule details."""
        j_ptr = [0] * NUM_JOBS
        m_end = [0] * NUM_MACHINES
        j_end = [0] * NUM_JOBS

        op_start = {}       # (job, op_idx) -> start time
        op_end = {}         # (job, op_idx) -> end time
        op_pos = {}         # (job, op_idx) -> position in code
        m_ops = [[] for _ in range(NUM_MACHINES)]  # ops on each machine in order

        for pos, job_id in enumerate(code):
            op_idx = j_ptr[job_id]
            m = JOB_M[job_id][op_idx]
            t = JOB_T[job_id][op_idx]

            start = j_end[job_id] if j_end[job_id] >= m_end[m] else m_end[m]
            end = start + t

            j_end[job_id] = end
            m_end[m] = end

            key = (job_id, op_idx)
            op_start[key] = start
            op_end[key] = end
            op_pos[key] = pos
            m_ops[m].append(key)

            j_ptr[job_id] += 1

        return max(m_end), op_start, op_end, op_pos, m_ops

    # ------------------------------------------------------------------
    # Critical path detection via backward pass
    # ------------------------------------------------------------------
    def find_critical_ops(self, code, op_start, op_end, op_pos, m_ops, makespan):
        """
        Backward pass on the disjunctive graph.
        An operation is critical if earliest_start == latest_start (zero slack).
        """
        # Build job successors: (job, k) → (job, k+1) for k < 9
        js = {}
        for j in range(NUM_JOBS):
            for k in range(OPS_PER_JOB - 1):
                js[(j, k)] = (j, k + 1)
        # Build machine successors from the actual schedule
        ms = {}
        for m in range(NUM_MACHINES):
            for i in range(len(m_ops[m]) - 1):
                ms[m_ops[m][i]] = m_ops[m][i + 1]
        # Map position → operation key
        pos_to_op = {pos: key for key, pos in op_pos.items()}
        latest_start = {}
        critical = set()
        # Process in reverse topological order (reverse of the code)
        for pos in range(TOTAL_OPS - 1, -1, -1):
            op = pos_to_op[pos]
            j, k = op
            t = JOB_T[j][k]
            # Latest end = min(latest_start of successors), or makespan if none
            candidates = []
            if op in js:
                candidates.append(latest_start[js[op]])
            if op in ms:
                candidates.append(latest_start[ms[op]])
            latest_end = min(candidates) if candidates else makespan
            ls = latest_end - t
            latest_start[op] = ls
            if abs(ls - op_start[op]) < 0.5:
                critical.add(op)
        return critical

    # ------------------------------------------------------------------
    # Constructive heuristics (dispatching rules)
    # ------------------------------------------------------------------
    def build_by_rule(self, rule='spt'):
        """
        Build a permutation by simulating the shop and picking the next
        operation using a dispatching rule.
        Rules: spt, lpt, mwkr, lwkr, est, fifo
        """
        j_ptr = [0] * NUM_JOBS
        m_end = [0] * NUM_MACHINES
        j_end = [0] * NUM_JOBS
        code = []
        unscheduled = set(range(NUM_JOBS))

        while unscheduled:
            ready = []
            for job_id in list(unscheduled):
                op = j_ptr[job_id]
                m = JOB_M[job_id][op]
                t = JOB_T[job_id][op]
                start = j_end[job_id] if j_end[job_id] >= m_end[m] else m_end[m]
                ready.append((job_id, op, m, t, start))

            if rule == 'spt':
                pick = min(ready, key=lambda x: x[3])
            elif rule == 'lpt':
                pick = max(ready, key=lambda x: x[3])
            elif rule == 'mwkr':
                pick = max(ready, key=lambda x: sum(JOB_T[x[0]][x[1]:]))
            elif rule == 'lwkr':
                pick = min(ready, key=lambda x: sum(JOB_T[x[0]][x[1]:]))
            elif rule == 'est':
                pick = min(ready, key=lambda x: x[4])
            else:  # fifo / default
                pick = ready[0]

            job_id = pick[0]
            code.append(job_id)
            op = j_ptr[job_id]
            m = JOB_M[job_id][op]
            t = JOB_T[job_id][op]
            start = j_end[job_id] if j_end[job_id] >= m_end[m] else m_end[m]
            j_end[job_id] = start + t
            m_end[m] = start + t
            j_ptr[job_id] += 1
            if j_ptr[job_id] == OPS_PER_JOB:
                unscheduled.remove(job_id)

        return code

    # ------------------------------------------------------------------
    # Critical-path local search (steepest descent on N5 neighbourhood)
    # ------------------------------------------------------------------
    def local_search(self, code, max_no_improve=300):
        """N5 neighbourhood: swap adjacent critical ops on the same machine."""
        best_code = list(code)
        best_ms, op_start, op_end, op_pos, m_ops = self.decode(best_code)
        critical = self.find_critical_ops(best_code, op_start, op_end, op_pos, m_ops, best_ms)

        stall = 0
        while stall < max_no_improve:
            improved = False
            best_neighbor = None
            best_neighbor_ms = best_ms

            for m in range(NUM_MACHINES):
                for i in range(len(m_ops[m]) - 1):
                    a, b = m_ops[m][i], m_ops[m][i + 1]
                    if a not in critical and b not in critical:
                        continue

                    pa, pb = op_pos[a], op_pos[b]
                    # Build neighbor by swapping the two positions
                    nb = list(best_code)
                    nb[pa], nb[pb] = nb[pb], nb[pa]
                    ms_nb = self.decode(nb)[0]

                    if ms_nb < best_neighbor_ms - 0.01:
                        best_neighbor_ms = ms_nb
                        best_neighbor = nb

            if best_neighbor is not None:
                best_code = best_neighbor
                best_ms = best_neighbor_ms
                _, op_start, op_end, op_pos, m_ops = self.decode(best_code)
                critical = self.find_critical_ops(best_code, op_start, op_end, op_pos, m_ops, best_ms)
                stall = 0
            else:
                stall += 1

        return best_code, best_ms

    # ------------------------------------------------------------------
    # Perturbation: random destruction + reconstruction
    # ------------------------------------------------------------------
    def perturb(self, code, d=2):
        """Remove d random jobs completely, then greedily reinsert them."""
        removed = random.sample(range(NUM_JOBS), d)
        partial = [j for j in code if j not in removed]

        for job_id in removed:
            for _ in range(OPS_PER_JOB):
                best_pos = 0
                best_ms = float('inf')
                for pos in range(len(partial) + 1):
                    cand = partial[:pos] + [job_id] + partial[pos:]
                    ms = self.decode(cand)[0]
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = pos
                partial = partial[:best_pos] + [job_id] + partial[best_pos:]

        return partial

    # ------------------------------------------------------------------
    # Main solver: ILS + Tabu
    # ------------------------------------------------------------------
    def solve(self):
        start_wall = time.time()
        best_code = None
        best_ms = float('inf')

        # ---- Phase 1: try all constructive rules + local search ----
        rules = ['spt', 'lpt', 'mwkr', 'lwkr', 'est']
        for rule in rules:
            if time.time() - start_wall > self.time_limit:
                break
            code = self.build_by_rule(rule)
            ms = self.decode(code)[0]
            if ms < best_ms:
                best_ms, best_code = ms, list(code)

            code, ms = self.local_search(code, max_no_improve=200)
            if ms < best_ms:
                best_ms, best_code = ms, list(code)

        # ---- Phase 2: Tabu Search with critical-path neighbourhood ----
        tabu = {}          # (op_a, op_b) → expiry iteration
        tenure = 12
        current_code = list(best_code)
        current_ms = best_ms
        iteration = 0
        restart_count = 0

        while time.time() - start_wall < self.time_limit:
            ms_val, op_start, op_end, op_pos, m_ops = self.decode(current_code)
            critical = self.find_critical_ops(current_code, op_start, op_end, op_pos, m_ops, ms_val)

            # Collect all N5 candidates
            candidates = []
            for m in range(NUM_MACHINES):
                for i in range(len(m_ops[m]) - 1):
                    a, b = m_ops[m][i], m_ops[m][i + 1]
                    if a not in critical and b not in critical:
                        continue
                    pa, pb = op_pos[a], op_pos[b]
                    candidates.append((a, b, pa, pb))

            # Evaluate all candidates (best improvement / best admissible)
            best_move = None
            best_move_code = None
            best_move_ms = float('inf')

            for a, b, pa, pb in candidates:
                tabu_key = (a, b) if a < b else (b, a)
                is_tabu = tabu.get(tabu_key, -1) >= iteration

                nb = list(current_code)
                nb[pa], nb[pb] = nb[pb], nb[pa]
                nb_ms = self.decode(nb)[0]

                # Aspiration: override tabu if new global best
                if is_tabu and nb_ms >= best_ms - 0.01:
                    continue

                if nb_ms < best_move_ms - 0.01:
                    best_move_ms = nb_ms
                    best_move_code = nb
                    best_move_key = tabu_key

            if best_move_code is None:
                # No admissible move — perturb and restart tabu
                current_code = self.perturb(current_code, d=random.randint(1, 3))
                current_ms = self.decode(current_code)[0]
                tabu.clear()
                restart_count += 1
                continue

            # Apply the move
            current_code = best_move_code
            current_ms = best_move_ms
            tabu[best_move_key] = iteration + tenure
            iteration += 1

            if current_ms < best_ms - 0.01:
                best_ms = current_ms
                best_code = list(current_code)

            # Periodic tabu list cleanup
            if iteration % 200 == 0:
                tabu = {k: v for k, v in tabu.items() if v >= iteration}

            # If stuck too long, perturb
            if iteration % 500 == 0:
                current_code = self.perturb(best_code, d=random.randint(1, 3))
                current_ms = self.decode(current_code)[0]
                tabu.clear()

        return best_ms, best_code


# ===================== Run =====================
if __name__ == "__main__":
    solver = JobShopSolver(time_limit=30)
    t0 = time.time()
    best_time, best_schedule = solver.solve()
    elapsed = time.time() - t0

    print("=" * 55)
    print("  10x10 Job Shop Scheduling -- Optimized Solver")
    print("=" * 55)
    print(f"  Best makespan:  {best_time}")
    print(f"  Solve time:     {elapsed:.1f}s")
    print(f"  Full schedule (100 ops):")
    print(f"  {best_schedule}")
    print("=" * 55)

    # ---- Correctness verification ----
    print("\n[Verification]")
    # 1. Each job must appear exactly 10 times
    from collections import Counter
    counts = Counter(best_schedule)
    ok = all(counts[j] == 10 for j in range(10))
    print(f"  Job counts (all=10): {dict(sorted(counts.items()))}  {'OK' if ok else 'FAIL'}")

    # 2. Re-decode and confirm makespan matches
    ms_check, starts, ends, _, m_ops = solver.decode(best_schedule)
    print(f"  Re-decoded makespan: {ms_check}  {'OK' if ms_check == best_time else 'MISMATCH'}")

    # 3. Check job precedence: for each job, op k must finish before op k+1 starts
    prec_ok = True
    for j in range(10):
        for k in range(9):
            if ends[(j, k)] > starts[(j, k + 1)]:
                print(f"  PRECEDENCE VIOLATION: job {j}, op {k} ends at {ends[(j,k)]} > op {k+1} starts at {starts[(j,k+1)]}")
                prec_ok = False
    print(f"  Job precedence: {'OK' if prec_ok else 'FAIL'}")

    # 4. Check machine capacity: no overlapping ops on same machine
    mach_ok = True
    for m in range(10):
        sorted_ops = sorted(m_ops[m], key=lambda x: starts[x])
        for i in range(len(sorted_ops) - 1):
            a, b = sorted_ops[i], sorted_ops[i + 1]
            if ends[a] > starts[b]:
                print(f"  MACHINE OVERLAP: machine {m}, {a} ends {ends[a]} > {b} starts {starts[b]}")
                mach_ok = False
    print(f"  Machine capacity: {'OK' if mach_ok else 'FAIL'}")

    # 5. Show per-machine schedule summary
    print(f"\n  Machine workloads (total time):")
    for m in range(10):
        total = sum(JOB_T[op[0]][op[1]] for op in m_ops[m])
        idle = best_time - total
        print(f"    M{m}: total_work={total}, idle={idle}")
