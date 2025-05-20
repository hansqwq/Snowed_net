import numpy as np
import time
import math
import random
import matplotlib.pyplot as plt
from Functions import zeroSupplement, inputTransform


class Network:
    def __init__(self, iteration, dimension=3):
        """
        构造 (N-1) 维雪花网络，其中 N = dimension + 1。
        例如，dimension=3 对应3维分形，即正四面体的4个顶点。
        iteration：总迭代次数（包含初始状态为第1层）
        """
        self.iteration = iteration - 1  # 除初始状态外的迭代次数
        self.N = dimension + 1  # 正 N-simplex 的顶点数
        self.network = {}  # 存储所有节点，键为 inputTransform 得到的字符串地址
        self.segCache = None  # 存储当前迭代的所有分段（整数数组）

    def buildNetwork(self):
        timeStart = time.time()
        # ----- 初始阶段 -----
        initial_nodes = []
        for i in range(1, self.N + 1):
            label = np.array([i, 0], dtype=int)
            key = inputTransform(list(map(str, label)))
            self.network[key] = label.copy()
            initial_nodes.append(label)
        self.segCache = np.array(initial_nodes)
        print("Iteration 1: Node Num:", len(self.network))

        # ----- 迭代生成 -----
        iter_count = 1
        while iter_count <= self.iteration:
            for seg in self.segCache:
                for i in range(1, self.N + 1):
                    new_label = np.concatenate((seg, [i, 0]))
                    key = inputTransform(list(map(str, new_label)))
                    self.network[key] = new_label.copy()
            newSegList = []
            for seg in self.segCache:
                r = seg[-2]
                for i in range(1, self.N + 1):
                    if i == r:
                        newSegList.append(np.append(seg, [i, 0]))
                    else:
                        newSegList.append(np.append(seg, [i, 1]))
                        newSegList.append(np.append(seg, [i, 2]))
            newSegList = np.array(newSegList)
            toDelete = []
            for idx, seg in enumerate(newSegList):
                if len(seg) >= 4 and seg[-4] == seg[-2]:
                    toDelete.append(idx)
            newSegList = np.delete(newSegList, toDelete, axis=0)
            self.segCache = newSegList
            print("Iteration %d: Total Node Num: %d" % (iter_count + 1, len(self.network)))
            iter_count += 1

        timeEnd = time.time()
        print("Network has been created. Time Consumed: %.2fs" % (timeEnd - timeStart))
        print("Total Node Num:", len(self.network))
        self.printNodesByLevel()

    def label_to_binary(self, label):
        """
        将节点的标签转换为完整的二进制地址（tier locator）。
        每对子标签占用 ⌈log₂ N⌉ + 1 位：
          - 固定前缀采用 N-1 的二进制表示（⌈log₂ N⌉ 位）；
          - 每对子标签：(a, b) 转换为 (a-1) 的二进制（⌈log₂ N⌉ 位）+ b（1位）。
        """
        bits_first = math.ceil(math.log2(self.N))
        extra = format(self.N - 1, f'0{bits_first}b')
        bin_address = extra
        for i in range(0, len(label), 2):
            a = label[i]
            b = label[i + 1]
            bin_a = format(a - 1, f'0{bits_first}b')
            bin_b = format(b, '01b')
            bin_address += bin_a + bin_b
        return bin_address

    def printNodesByLevel(self):
        """
        按子标签对数量分层，只打印层级 1~3 的节点信息。
        """
        level_dict = {}
        for key, label in self.network.items():
            level = len(label) // 2
            if level not in level_dict:
                level_dict[level] = []
            level_dict[level].append((label, self.label_to_binary(label)))
        print("\n===== 每层节点信息 (仅显示前三层) =====")
        for lvl in sorted(level_dict.keys()):
            if lvl > 3:
                continue
            print(f"Level {lvl} (共 {len(level_dict[lvl])} 个节点):")
            for lab, bin_addr in level_dict[lvl]:
                print("  Label:", lab, "-> Binary:", bin_addr)
            print("-------------------------")

    def routePathNodes(self, source_label, dest_label):
        """
        计算从源节点到目标节点的路由过程，返回路径上经过的各节点（以列表形式表示）。
        算法步骤：
          1. 将 source_label 与 dest_label 拆分为对子列表。
          2. 找出最长公共对子序列，得到公共祖先的 label。
          3. 若没有公共对子（即 common == 0），则：
             - 将源节点的初始节点作为上行终点；
             - 将目标节点的初始节点作为下行起点；
             - 路由路径为：源节点上行至初始节点，再从初始节点下行至目标节点。
          4. 否则：
             - 上行路径：从源节点逐步去掉最后一对子标签，直到达到公共祖先；
             - 下行路径：从公共祖先开始，依次添加 dest_label 中剩余的对子。
          5. 合并上行与下行，公共祖先只出现一次，返回完整路径（所有节点均转换为 Python 列表）。
        """

        def label_to_pairs(label):
            return [tuple(label[i:i + 2]) for i in range(0, len(label), 2)]

        src_pairs = label_to_pairs(source_label)
        dst_pairs = label_to_pairs(dest_label)
        # 找出最长公共对子序列
        common = 0
        for a, b in zip(src_pairs, dst_pairs):
            if a == b:
                common += 1
            else:
                break
        if common == 0:
            # 若没有公共对子，则采用源和目标的初始节点作为上行终点和下行起点
            s_init = source_label[0:2].tolist()
            d_init = dest_label[0:2].tolist()
            common_label = s_init
            # 上行路径：从源节点向上返回到源初始节点
            upward = []
            curr = source_label.copy()
            while len(curr) // 2 > 1:
                upward.append(curr.copy().tolist())
                curr = curr[:-2]
            upward.append(s_init)
            # 下行路径：从目标初始节点向下扩展至目标节点
            downward = []
            curr = d_init.copy()
            for pair in dst_pairs[1:]:
                curr.extend(pair)
                downward.append(curr.copy())
            # 构造完整路径：上行路径 + 下行路径，中间采用初始层的连通性实现跳转
            route_nodes = upward + [d_init] + downward
        else:
            common_label = []
            for i in range(common):
                common_label.extend(src_pairs[i])
            # 上行路径
            upward = []
            curr = source_label.copy()
            while len(curr) // 2 > common:
                upward.append(curr.copy().tolist())
                curr = curr[:-2]
            upward.append(common_label)
            # 下行路径
            downward = []
            curr = common_label.copy()
            for pair in dst_pairs[common:]:
                curr.extend(pair)
                downward.append(curr.copy())
            if downward:
                route_nodes = upward + downward
            else:
                route_nodes = upward
        return route_nodes

    def route_interactive(self):
        """
        交互式路由测试：
        用户输入源节点和目标节点地址，显示完整的路由路径（节点 label）。
        """
        src_input = list(input("Enter source node address (例如: 4 0 3 2 4 0): ").split())
        dst_input = list(input("Enter destination node address (例如: 2 0 2 0): ").split())
        src_key = inputTransform(src_input)
        dst_key = inputTransform(dst_input)
        if src_key not in self.network or dst_key not in self.network:
            print("源节点或目标节点未找到！")
            return
        src_label = self.network[src_key]
        dst_label = self.network[dst_key]
        path_nodes = self.routePathNodes(src_label, dst_label)
        print("Routing Process (node labels):")
        for node in path_nodes:
            print("  ", node)

    def simulate_routing(self, num_tests=100):
        """
        对网络中随机选取源与目标节点进行路由仿真，
        统计平均路由计算时间和平均路径跳数，
        确保每次随机选择的源与目标节点不同。
        """
        keys = list(self.network.keys())
        total_time = 0.0
        path_lengths = []
        for _ in range(num_tests):
            src_key = random.choice(keys)
            dst_key = random.choice(keys)
            while dst_key == src_key:
                dst_key = random.choice(keys)
            src_label = self.network[src_key]
            dst_label = self.network[dst_key]
            start = time.time()
            path = self.routePathNodes(src_label, dst_label)
            elapsed = time.time() - start
            total_time += elapsed
            path_lengths.append(len(path))
        avg_time = total_time / num_tests
        avg_path_length = sum(path_lengths) / num_tests
        print("\n===== 路由性能仿真 =====")
        print("平均路由计算时间: {:.6f} 秒".format(avg_time))
        print("平均路径跳数: {:.2f}".format(avg_path_length))
        return avg_time, avg_path_length, len(self.network)


def performance_visualization(dimension, iterations_list, num_tests=100):
    """
    对不同迭代次数下的网络进行仿真，统计网络规模、平均路由计算时间和平均路径跳数，
    并分别绘制三个独立的图表，确保图表布局无重叠且横坐标刻度清晰。
    """
    avg_times = []
    avg_hops = []
    total_nodes = []

    for iters in iterations_list:
        print("\n---------- 迭代次数: {} ----------".format(iters))
        net = Network(iters, dimension)
        net.buildNetwork()
        avg_time, avg_path_length, node_count = net.simulate_routing(num_tests)
        avg_times.append(avg_time)
        avg_hops.append(avg_path_length)
        total_nodes.append(node_count)

    # 图表1：网络规模
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_list, total_nodes, marker='o')
    plt.xlabel("Iteration Count")
    plt.ylabel("Total Node Count")
    plt.title("Network Scale")
    plt.xticks(iterations_list, rotation=45)
    plt.tight_layout()
    plt.show()

    # 图表2：平均路由计算时间
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_list, avg_times, marker='o', color='r')
    plt.xlabel("Iteration Count")
    plt.ylabel("Avg Routing Time (sec)")
    plt.title("Routing Computation Time")
    plt.xticks(iterations_list, rotation=45)
    plt.tight_layout()
    plt.show()

    # 图表3：平均路径跳数
    plt.figure(figsize=(8, 6))
    plt.plot(iterations_list, avg_hops, marker='o', color='g')
    plt.xlabel("Iteration Count")
    plt.ylabel("Avg Path Hops")
    plt.title("Routing Path Length")
    plt.xticks(iterations_list, rotation=45)
    plt.tight_layout()
    plt.show()


def interactive_mode():
    """
    交互模式：用户输入迭代次数与分形维度构建网络，
    打印前三层节点信息，再输入源节点与目标节点，显示路由路径。
    """
    iteration = int(input("Enter number of iterations: "))
    dimension = int(input("Enter dimension: "))
    net = Network(iteration, dimension)
    net.buildNetwork()
    net.printNodesByLevel()
    net.route_interactive()


def simulation_mode():
    """
    仿真模式：用户输入分形维度和多个迭代次数，
    自动构建网络并仿真测试路由性能，生成独立的性能图表。
    """
    dimension = int(input("Enter dimension: "))
    iteration_input = input("Enter iteration counts (comma separated, e.g., 2,3,4,5): ")
    iterations_list = [int(x.strip()) for x in iteration_input.split(",")]
    num_tests = int(input("Enter number of tests for each iteration count: "))
    performance_visualization(dimension, iterations_list, num_tests)


def fault_simulation_mode():
    """
    故障仿真模式：
    用户输入分形维度、迭代次数和故障率（节点失效比例），以及测试次数，
    构建网络后随机移除一定比例的节点，更新网络为仅健康节点，
    然后在健康网络上执行路由算法，利用分形连通性验证路由，
    输出故障仿真成功率、成功路由的平均路由计算时间和平均路径跳数。
    """
    dimension = int(input("Enter dimension: "))
    iteration = int(input("Enter number of iterations: "))
    fault_rate = float(input("Enter fault rate (e.g., 0.1 表示10%的节点失效): "))
    num_tests = int(input("Enter number of tests for fault simulation: "))

    # 构建完整网络
    net = Network(iteration, dimension)
    net.buildNetwork()
    total_nodes = len(net.network)
    print("Original Total Node Count:", total_nodes)

    # 模拟节点故障：随机移除 fault_rate 比例的节点
    healthy_count = int(total_nodes * (1 - fault_rate))
    healthy_keys = random.sample(list(net.network.keys()), healthy_count)
    healthy_network = {k: net.network[k] for k in healthy_keys}
    print("Healthy Node Count after fault simulation:", len(healthy_network))

    # 更新网络为健康网络
    net.network = healthy_network

    total_time = 0.0
    path_lengths = []
    successful = 0
    for _ in range(num_tests):
        keys = list(healthy_network.keys())
        src_key = random.choice(keys)
        dst_key = random.choice(keys)
        while dst_key == src_key:
            dst_key = random.choice(keys)
        src_label = healthy_network[src_key]
        dst_label = healthy_network[dst_key]
        start = time.time()
        path = net.routePathNodes(src_label, dst_label)
        elapsed = time.time() - start
        # 判断路由成功的条件：路径的最后一个节点应等于目标节点的列表表示
        if path and path[-1] == dst_label.tolist():
            successful += 1
            total_time += elapsed
            path_lengths.append(len(path))
    success_rate = successful / num_tests if num_tests > 0 else 0
    avg_time = total_time / successful if successful > 0 else 0
    avg_path_length = sum(path_lengths) / successful if successful > 0 else 0

    print("\n===== 故障仿真结果 =====")
    print("Fault Rate: {:.0%}".format(fault_rate))
    print("Routing Success Rate: {:.2%}".format(success_rate))
    print("Average Routing Time (successful routes): {:.6f} 秒".format(avg_time))
    print("Average Path Hops (successful routes): {:.2f}".format(avg_path_length))


def main_mode():
    mode = input("Select mode (1: Interactive Mode, 2: Simulation Mode, 3: Fault Simulation Mode): ")
    if mode.strip() == "1":
        interactive_mode()
    elif mode.strip() == "2":
        simulation_mode()
    elif mode.strip() == "3":
        fault_simulation_mode()
    else:
        print("无效的模式选择！")


if __name__ == "__main__":
    main_mode()