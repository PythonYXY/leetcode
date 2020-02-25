

# 3. 数组中的重复数字

```
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。
```

## 代码
```java
public class Solution {
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (length == 0) return false;
        
        for (int index = 0; index < length; index++) {
            while (numbers[index] != index) {
                if (numbers[index] == numbers[numbers[index]]) {
                    duplication[0] = numbers[index];
                    return true;
                }
                swap(numbers, index, numbers[index]);
            }
        }
        
        return false;
    }
    
    private void swap(int[] numbers, int a, int b) {
        int temp = numbers[a];
        numbers[a] = numbers[b];
        numbers[b] = temp;
    }
}
```

# L287. 寻找重复数

## 题目描述

```
给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。

示例 1:

输入: [1,3,4,2,2]
输出: 2
示例 2:

输入: [3,1,3,4,2]
输出: 3
说明：

不能更改原数组（假设数组是只读的）。
只能使用额外的 O(1) 的空间。
时间复杂度小于 O(n2) 。
数组中只有一个重复的数字，但它可能不止重复出现一次。
```

## 思路1（二分法）


这道题的关键是对要定位的“数”做二分，而不是对数组的索引做二分。要定位的“数”根据题意在 1 和 n 之间，每一次二分都可以将搜索区间缩小一半。

## 代码1
```java
public class Solution {

    public int findDuplicate(int[] nums) {
        int len = nums.length;
        int left = 1;
        int right = len - 1;
        while (left < right) {
            // int mid = left + (right - left) / 2;
            int mid = (left + right) >>> 1;
            int counter = 0;
            for (int num : nums) {
                if (num <= mid) {
                    counter += 1;
                }
            }
            if (counter > mid) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
}

```

## 思路2（快慢指针）

将数组内的元素值看作是该节点指向的下个节点在数组中的索引值，这样就可以将数组转化为链表进行处理。由于元素值是从1开始，而索引值是从0开始，因此链表的长度为最大元素值加1，也就是说链表中一定存在两个节点有着相同的值，它们指向同一个节点，也即链表中有环的存在。因此本题可以转化为寻找链表中环入口的问题。具体可参考题目环形链表 II的解法：



设置快慢指针从A1出发。假设slow和fast在A3相遇，根据以上公式（fast在与slow相遇前走过了k圈）。head从A1出发，slow从A3出发，经过a步以后，两点将在A2（环的入口）相遇。

## 代码2

```java
public class Solution {

    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        
        while (slow != fast || fast == 0) {
            slow = nums[slow];
            fast = nums[nums[fast]];
        }
        
        int res = 0;
        while (res != slow) {
            res = nums[res];
            slow = nums[slow];
        }
        
        return res;
    }
}

```

# 4.二维数组中的查找
## L240. 搜索二维矩阵 II

```
编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：

每行的元素从左到右升序排列。
每列的元素从上到下升序排列。
示例:

现有矩阵 matrix 如下：

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
给定 target = 5，返回 true。

给定 target = 20，返回 false。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-a-2d-matrix-ii
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
public class Solution {
    /**
    1 10 20 30
    2 11 21 31
    3 12 22 32
    4 13 23 33
    
    观察例子，发现与右上角的元素同一行的元素都不大于它，而同一列的元素都不小于它
    **/
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix.length == 0 || matrix[0].length == 0) return false;

        int row = 0, col = matrix[0].length - 1;

        while (row < matrix.length && col >= 0) {
            if (target == matrix[row][col]) return true;
            if (target < matrix[row][col]) {
                col--;
            } else {
                row++;
            }
        }

        return false;
    }
}
```

## L74. 搜索二维矩阵

```编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

每行中的整数从左到右按升序排列。
每行的第一个整数大于前一行的最后一个整数。
示例 1:

输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
输出: true
示例 2:

输入:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
输出: false

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/search-a-2d-matrix
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class Solution {
    public boolean searchMatrix(int[][] matrix, int target) {
        
        int lrow = 0, rrow = matrix.length - 1;
        if (rrow == -1 || matrix[0].length == 0) return false;
        while (lrow != rrow) {
            int mid = (lrow + rrow + 1) >>> 1;

            if (target < matrix[mid][0]) {
                rrow = mid - 1;
            } else {
                lrow = mid;
            }
        }

        if (matrix[lrow][0] == target) return true;
        

        int lcol = 0, rcol = matrix[lrow].length - 1;

        while (lcol != rcol) {
            int mid = (lcol + rcol) >>> 1;

            if (target > matrix[lrow][mid]) {
                lcol = mid + 1;
            } else {
                rcol = mid;
            }
        }


        return matrix[lrow][lcol] == target;
    }
}
```

# 8.二叉树的下一个节点
```
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
```

## 代码
```java
/*
public class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;

    TreeLinkNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode pNode)
    {
        if (pNode == null) return null;
        if (pNode.right != null) {
            TreeLinkNode node = pNode.right;
            while (node.left != null) node = node.left;
            return node;
        }
        
        TreeLinkNode par = pNode.next;
        while (par != null) {
            if (par.left == pNode) return par;
            pNode = par;
            par = par.next;
        }
        
        return null;
    }
}
```

# 9. 用两个栈实现队列
## L232. 用栈实现队列
```
使用栈实现队列的下列操作：

push(x) -- 将一个元素放入队列的尾部。
pop() -- 从队列首部移除元素。
peek() -- 返回队列首部的元素。
empty() -- 返回队列是否为空。
示例:

MyQueue queue = new MyQueue();

queue.push(1);
queue.push(2);  
queue.peek();  // 返回 1
queue.pop();   // 返回 1
queue.empty(); // 返回 false
说明:

你只能使用标准的栈操作 -- 也就是只有 push to top, peek/pop from top, size, 和 is empty 操作是合法的。
你所使用的语言也许不支持栈。你可以使用 list 或者 deque（双端队列）来模拟一个栈，只要是标准的栈操作即可。
假设所有操作都是有效的 （例如，一个空的队列不会调用 pop 或者 peek 操作）。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/implement-queue-using-stacks
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码

```java
class MyQueue {

    Stack<Integer> stack1, stack2;
    /** Initialize your data structure here. */
    public MyQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }
    
    /** Push element x to the back of queue. */
    public void push(int x) {
        stack1.push(x);
    }
    
    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        int ret = peek();
        stack2.pop();
        return ret;
    }
    
    /** Get the front element. */
    public int peek() {
        if (!stack2.empty()) {
            return stack2.peek();
        }

        while (!stack1.empty()) {
            stack2.push(stack1.pop());
        }
        return stack2.peek();
    }
    
    /** Returns whether the queue is empty. */
    public boolean empty() {
        return stack1.empty() && stack2.empty();
    }
}

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue obj = new MyQueue();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.peek();
 * boolean param_4 = obj.empty();
 */
```


## L225. 用队列实现栈
```
使用队列实现栈的下列操作：

push(x) -- 元素 x 入栈
pop() -- 移除栈顶元素
top() -- 获取栈顶元素
empty() -- 返回栈是否为空
注意:

你只能使用队列的基本操作-- 也就是 push to back, peek/pop from front, size, 和 is empty 这些操作是合法的。
你所使用的语言也许不支持队列。 你可以使用 list 或者 deque（双端队列）来模拟一个队列 , 只要是标准的队列操作即可。
你可以假设所有操作都是有效的（例如, 对一个空的栈不会调用 pop 或者 top 操作）。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/implement-stack-using-queues
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码

```java
class MyStack {

    Queue<Integer> que1, que2;

    /** Initialize your data structure here. */
    public MyStack() {
        que1 = new LinkedList<>();
        que2 = new LinkedList<>();    
    }
    
    /** Push element x onto stack. */
    public void push(int x) {
        if (que1.size() == 0 && que2.size() == 0) que1.add(x);
        else if (que1.size() != 0) que1.add(x);
        else que2.add(x);
    }
    
    /** Removes the element on top of the stack and returns that element. */
    public int pop() {
        int ret;
        if (que1.size() == 0) {
            while (que2.size() != 1) que1.add(que2.poll());
            ret = que2.poll();
        } else {
            while (que1.size() != 1) que2.add(que1.poll());
            ret = que1.poll();
        }
        return ret;
    }
    
    /** Get the top element. */
    public int top() {
        int ret;
        if (que1.size() == 0) {
            while (que2.size() != 1) que1.add(que2.poll());
            ret = que2.poll();
            que1.add(ret);
        } else {
            while (que1.size() != 1) que2.add(que1.poll());
            ret = que1.poll();
            que2.add(ret);
        }
        return ret;
    }
    
    /** Returns whether the stack is empty. */
    public boolean empty() {
        return que1.size() == 0 && que2.size() == 0;
    }
}

/**
 * Your MyStack object will be instantiated and called as such:
 * MyStack obj = new MyStack();
 * obj.push(x);
 * int param_2 = obj.pop();
 * int param_3 = obj.top();
 * boolean param_4 = obj.empty();
 */
```

## 11：旋转数组的最小数字
# L153. 寻找旋转排序数组中的最小值

## 题目描述

```
假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

请找出其中最小的元素。

你可以假设数组中不存在重复元素。

示例 1:

输入: [3,4,5,1,2]
输出: 1
示例 2:

输入: [4,5,6,7,0,1,2]
输出: 0
```

## 思路

使用二分法可以将时间复杂度缩减到O(logn)。

这里排除中位数的逻辑是如果mid上的值大于right上的值，则最小元素一定处于右区间（不包含mid）。
## 代码

```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
    
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return nums[left];
    }
}
```

# L154. 寻找旋转排序数组中的最小值 II

## 题目描述

```
假设按照升序排序的数组在预先未知的某个点上进行了旋转。

( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

请找出其中最小的元素。

注意数组中可能存在重复的元素。

示例 1：

输入: [1,3,5]
输出: 1
示例 2：

输入: [2,2,2,0,1]
输出: 0
说明：

这道题是 寻找旋转排序数组中的最小值 的延伸题目。
允许重复会影响算法的时间复杂度吗？会如何影响，为什么？

```

## 思路

在153的基础上增加了重复元素的情况。这里需要额外处理中间元素与末尾元素相同的情况。这种情况下无法判断最小元素是在左区间还是右区间，例如：[2, 0, 1, 2, 2, 2, 2]与[2, 2, 2, 2, 0, 1, 2]，因此只能通过减少right来缩小区间进行进一步的搜索。该算法在最坏情况下的时间复杂度为O(n)。

## 代码

```java
class Solution {
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
    
        while (left < right) {
            int mid = (left + right) >>> 1;
            if (nums[mid] == nums[right]) {
                right--;
            } else if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return nums[left];
    }
}
```

# 14. 剪梯子
```
给你一根长度为n的绳子，请把绳子剪成整数长的m段（m、n都是整数，n>1并且m>1），每段绳子的长度记为k[0],k[1],...,k[m]。请问k[0]xk[1]x...xk[m]可能的最大乘积是多少？例如，当绳子的长度是8时，我们把它剪成长度分别为2、3、3的三段，此时得到的最大乘积是18。
输入描述:
输入一个数n，意义见题面。（2 <= n <= 60）
输出描述:
输出答案。
示例1
输入
复制
8
输出
复制
18
```

## 代码
```java
public class Solution {
    public int cutRope(int target) {
        int[] dp = new int[target + 1];
        dp[1] = dp[2] = 1;
        for (int i = 3; i <= target; i++) {
            for (int j = 1; j < i; j++) {
                dp[i] = Math.max(dp[i], Math.max(dp[j], j) * Math.max(dp[i - j], i - j));
            }
        }
        
        return dp[target];
    }
}


```

# 15. 二进制中1的个数
# L191. 位1的个数
```
编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。

 

示例 1：

输入：00000000000000000000000000001011
输出：3
解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
示例 2：

输入：00000000000000000000000010000000
输出：1
解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
示例 3：

输入：11111111111111111111111111111101
输出：31
解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
 

提示：

请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。
 

进阶:
如果多次调用这个函数，你将如何优化你的算法？

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/number-of-1-bits
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
public class Solution {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        int count = 0;
        while (n != 0) {
            count++;
            n = n & (n - 1);
        }

        return count;
    }
}
```

## 16. 数值的整数次方
## L50. Pow(x, n)
```
实现 pow(x, n) ，即计算 x 的 n 次幂函数。

示例 1:

输入: 2.00000, 10
输出: 1024.00000
示例 2:

输入: 2.10000, 3
输出: 9.26100
示例 3:

输入: 2.00000, -2
输出: 0.25000
解释: 2-2 = 1/22 = 1/4 = 0.25
说明:

-100.0 < x < 100.0
n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/powx-n
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class Solution {
    // 注意测试用例2.00000, -2147483648
    public double myPow(double x, int n) {
        long N = (long)n;
        x = N < 0 ? 1 / x : x;
        N = Math.abs(N);

        if (N == 0) return 1;
        if (N == 1) return x;

        double ret = myPow(x, (int)(N >> 1));

        if ((N & 1) == 1) ret *= ret * x;
        else ret *= ret;

        return ret;
    }
}
```

## 面试题17. 打印从1到最大的n位数
```
输入数字 n，按顺序打印出从 1 到最大的 n 位十进制数。比如输入 3，则打印出 1、2、3 一直到最大的 3 位数 999。

示例 1:

输入: n = 1
输出: [1,2,3,4,5,6,7,8,9]
 

说明：

用返回一个整数列表来代替打印
n 为正整数

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/da-yin-cong-1dao-zui-da-de-nwei-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class Solution {

    public int[] printNumbers(int n) {
        ArrayList<Integer> ret = new ArrayList<>();
        int[] cur = new int[n];

        helper(ret, cur, n, 0);
        return ret.stream().mapToInt(i -> i).toArray();
    }

    public void helper(ArrayList<Integer> ret, int[] cur, int n, int index) {
        if (index >= n) {
            int num = arrayToInt(cur);
            if (num != 0) ret.add(arrayToInt(cur));
            return;
        }

        for (int i = 0; i <= 9; i++) {
            cur[index] = i;
            helper(ret, cur, n, index + 1);
        }
    }

    public int arrayToInt(int[] arr) {
        int res = 0;
        for (int i = 0; i < arr.length; i++) {
            res = res * 10 + arr[i];
        }
        return res;
    }
}
```

## 18. 删除链表中的重复元素
```
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
```

## 代码
```java
/*
 public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    public ListNode deleteDuplication(ListNode pHead)
    {
        ListNode dummy = new ListNode(-1);
        dummy.next = pHead;
        ListNode cur = pHead;
        ListNode pre = dummy;

        while (cur != null) {
            if (cur.next != null && cur.val == cur.next.val) {
                int dupNum = cur.val;
                while (cur.val == dupNum) {
                    pre.next = cur.next;
                    cur = pre.next;
                    if (cur == null) return dummy.next;
                }
            } else {
                pre = cur;
                cur = cur.next;
            }
        }

        return dummy.next;

    }
}
```

# 19. 正则表达式匹配
```
给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

'.' 匹配任意单个字符
'*' 匹配零个或多个前面的那一个元素
所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
示例 1:

输入:
s = "aa"
p = "a"
输出: false
解释: "a" 无法匹配 "aa" 整个字符串。
示例 2:

输入:
s = "aa"
p = "a*"
输出: true
解释: 因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
示例 3:

输入:
s = "ab"
p = ".*"
输出: true
解释: ".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
示例 4:

输入:
s = "aab"
p = "c*a*b"
输出: true
解释: 因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
示例 5:

输入:
s = "mississippi"
p = "mis*is*p*."
输出: false

```

## 代码（DP）

```java
class Solution {
    public boolean isMatch(String s, String p) {
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;

        for (int i = 0; i <= s.length(); i++) {
            for (int j = 1; j <= p.length(); j++) {
                if (p.charAt(j - 1) == '*') {
                    dp[i][j] = dp[i][j - 2] || (i > 0 && (s.charAt(i - 1) == p.charAt(j - 2) || p.charAt(j - 2) == '.') && dp[i - 1][j]);
                } else {
                    dp[i][j] = i > 0 && (s.charAt(i - 1) == p.charAt(j - 1) || p.charAt(j - 1) == '.') && dp[i - 1][j - 1];
                }
            }
        }

        return dp[s.length()][p.length()];

    }
}
```

