import numpy as np
import random
from typing import List, Optional, Tuple
import math
import time

# 全局参数定义
GOAL_RADIUS = 5.0          # 目标区域半径
DELTA_T = 0.1             # 时间步长
MAX_ITER = 1000          # 最大规划迭代次数
BIAS_DECAY_RATE = 0.95  # 采样偏置衰减率
DELTA = 1.0             # 到达路标点触发的更新距离阈值
MERGE_DISTANCE = 2.0   # 子树合并的距离阈值
P_COLLISION_THRESHOLD = 0.1  # 碰撞概率阈值

# 全局变量，用于存储启发式区域和相关信息
heuristic_region = None  # 当前启发式区域
waypoints = []           # 路标点列表
bias = 1.0              # 启发式采样偏置
node_list = []          # 已到达的路标点列表
last_node = None        # 上一个触发更新的路标点

class Node:
    """节点类，存储时间基树中的节点信息"""
    def __init__(self, state: np.ndarray, parent: Optional['Node'] = None, t: float = 0.0, control: np.ndarray = None):
        self.state = np.array(state)      # 节点状态 [x, y, theta]
        self.parent = parent             # 父节点
        self.children = []              # 子节点列表
        self.depth = 0 if parent is None else parent.depth + 1  # 节点深度
        self.t = t                      # 时间戳
        self.control = control if control is not None else np.array([0.0, 0.0])  # 控制输入 [v, w]
        self.p_collision = 0.0         # 碰撞概率

    def add_child(self, child: 'Node'):
        """添加子节点"""
        self.children.append(child)

class Tree:
    """树结构类"""
    def __init__(self, root: Node):
        self.root = root     # 树的根节点
        self.nodes = [root] # 包含所有节点

    def add_node(self, node: Node):
        """添加节点到树中"""
        self.nodes.append(node)

def in_goal_region(goal ,current_state,goal_radis):
    current_state = np.arrat(current_state)
    goal = np.array(goal)
    dx ,dy = goal - current_state
    distance = math.hypot(dx , dy)
    return distance <= goal_radis
    

def neural_sample(current_state: np.ndarray, goal_state: np.ndarray, map_info) -> np.ndarray:
    """神经自适应引导采样函数（算法2）"""
    global heuristic_region, waypoints, bias, node_list, last_node
    
    # 如果还没有初始化启发式区域，则进行初始推理
    if heuristic_region is None:
        heuristic_region, waypoints = net_infer(None, goal_state, map_info)
        bias = 1.0
        node_list = []
        last_node = None
    
    # 检查是否接近路标点，需要更新启发式区域
    for waypoint in waypoints:
        if np.linalg.norm(current_state[:2] - waypoint.state[:2]) < DELTA:
            node_list.append(waypoint)
            last_node = waypoint
            # 当接近路标点时，更新启发式区域并重置采样偏置
            heuristic_region, waypoints = net_infer(last_node, goal_state, map_info)
            bias = 1.0
            break
    
    # 自适应采样：根据偏置在启发式区域内采样或进行随机采样
    if random.random() < bias:
        # 在启发式区域内进行贪婪采样
        min_xy, max_xy = heuristic_region
        x = random.uniform(min_xy[0], max_xy[0])
        y = random.uniform(min_xy[1], max_xy[1])
        theta = random.uniform(0, 2 * np.pi)
        x_rand = np.array([x, y, theta])
    else:
        # 在整个状态空间中进行随机采样
        x_rand = np.array([random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 2 * np.pi)])
    
    # 如果采样没有显著进展，衰减采样偏置以增加随机采样的可能性
    bias = bias * BIAS_DECAY_RATE
    return x_rand

def net_infer(last_node: Optional[Node], goal_state: np.ndarray, map_info) -> Tuple[np.ndarray, List[Node]]:





def find_closest_tree(x_rand: np.ndarray, trees: List[Tree]) -> Tree:
    """找到距离随机采样点最近的树"""
    min_cost = float('inf')
    closest_tree = None
    
    for tree in trees:
        for node in tree.nodes:
            # 计算节点到随机采样点的代价
            cost_value = np.linalg.norm(node.state[:2] - x_rand[:2])
            if cost_value < min_cost:
                min_cost = cost_value
                closest_tree = tree
    return closest_tree

def meet(root_tree: Tree, sub_trees: List[Tree]) -> bool:
    """检查主树是否与任何子树相连"""
    for sub_tree in sub_trees:
        for root_node in root_tree.nodes:
            for sub_node in sub_tree.nodes:
                if np.linalg.norm(root_node.state[:2] - sub_node.state[:2]) < MERGE_DISTANCE:
                    return True
    return False

def multi_search(x_rand: np.ndarray, closest_tree: Tree, sub_trees: List[Tree], goal_state: np.ndarray) -> List[Tree]:
    close_trees = []
    
    # 找出所有距离采样点较近的子树
    for tree in sub_trees:
        min_distance = min(np.linalg.norm(node.state[:2] - x_rand[:2]) for node in tree.nodes)
        if min_distance < MERGE_DISTANCE:
            close_trees.append(tree)
    
    if len(close_trees) == 1:
        # 只有一个最近的子树，将新节点添加到该子树
        nearest_node = min(close_trees[0].nodes, key=lambda n: np.linalg.norm(n.state[:2] - x_rand[:2]))
        new_node = Node(x_rand, parent=nearest_node)
        nearest_node.add_child(new_node)
        close_trees[0].add_node(new_node)
    elif len(close_trees) > 1:
        # 多个子树足够接近，进行子树合并
        base_tree = close_trees[0]
        for other_tree in close_trees[1:]:
            base_tree.root.add_child(other_tree.root)
            base_tree.nodes.extend(other_tree.nodes)
            sub_trees.remove(other_tree)
    else:
        # 没有足够接近的子树，创建新的子树
        new_root = Node(x_rand)
        new_tree = Tree(new_root)
        sub_trees.append(new_tree)
    
    return sub_trees

def risk_grow(tree: Tree, x_rand: np.ndarray, map_info) -> bool:
    """风险感知增长函数"""
    # 找到距离采样点最近的节点
    nearest_node = min(tree.nodes, key=lambda n: np.linalg.norm(n.state[:2] - x_rand[:2]))
    
    # 计算控制输入并生成新状态
    control = np.array([1.0, 0.0])  # 简单的控制输入
    new_state = nearest_node.state + control * DELTA_T  # 简化的运动学模型
    
    new_node = Node(new_state, parent=nearest_node, t=nearest_node.t + DELTA_T, control=control)
    
    # 计算碰撞概率并进行碰撞检测
    new_node.p_collision = 0.0  # 需要实现具体的碰撞概率计算
    
    # 如果碰撞概率在可接受范围内且无碰撞，则添加新节点
    if new_node.p_collision < P_COLLISION_THRESHOLD:
        nearest_node.add_child(new_node)
        tree.add_node(new_node)
        return True
    
    return False

def namr_rrt(start_state: np.ndarray, goal_state: np.ndarray, map_info) -> Tree:
    """NAMR-RRT主算法（算法1）"""
    # 初始化主树
    root_node = Node(start_state, t=time.time())
    root_tree = Tree(root_node)
    
    # 初始化以目标点为根的子树
    goal_node = Node(goal_state)
    sub_trees = [Tree(goal_node)]
    
    iteration = 0
    while not in_goal_region(root_tree.root.state, goal_state) and iteration < MAX_ITER:
        if not meet(root_tree, sub_trees):
            # 主树与子树未连接，进行采样和扩展
            x_rand = neural_sample(root_tree.root.state, goal_state, map_info)
            all_trees = [root_tree] + sub_trees
            closest_tree = find_closest_tree(x_rand, all_trees)
            
            if closest_tree == root_tree:
                # 最近的树是主树，向主树扩展
                risk_grow(root_tree, x_rand, map_info)
            else:
                # 最近的树是子树，向子树扩展
                sub_trees = multi_search(x_rand, closest_tree, sub_trees, goal_state)
        else:
            # 主树与子树已连接，使用子树引导主树的生长
            connected_sub_tree = next((t for t in sub_trees if meet(root_tree, [t])), None)
            if connected_sub_tree:
                x_rand = random.choice(connected_sub_tree.nodes).state
                risk_grow(root_tree, x_rand, map_info)
        
        iteration += 1
    
    return root_tree

# 使用示例
if __name__ == "__main__":
    start_state = np.array([0.0, 0.0, 0.0])
    goal_state = np.array([20.0, 20.0, 0.0])
    
   
    map_obj = None 
    
    # 执行NAMR-RRT规划
    path_tree = namr_rrt(start_state, goal_state, map_obj)
    
    if path_tree:
        print("路径规划完成")
        print(f"生成的树包含 {len(path_tree.nodes)} 个节点")
    else:
        print("路径规划失败")