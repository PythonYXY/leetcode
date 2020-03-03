

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

# 20.表示数值的字符串
```
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100"、"5e2"、"-123"、"3.1416"、"0123"及"-1E-16"都表示数值，但"12e"、"1a3.14"、"1.2.3"、"+-5"及"12e+5.4"都不是。

 

注意：本题与主站 65 题相同：https://leetcode-cn.com/problems/valid-number/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/biao-shi-shu-zhi-de-zi-fu-chuan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class Solution {
    private int index = 0;
    public boolean isNumber(String s) {
        char[] sArr = s.toCharArray();
        while(index < sArr.length && sArr[index] == ' ')
            index++;
        boolean res = scanNum(sArr);
        if(index == sArr.length){
            return res;
        }
        if(sArr[index] == '.'){
            index++;
            res = scanUnsignedNum(sArr) || res;
            //wdnmd短路执行！！！！！！！！！！！！！！！！！！
        }
        if(index == sArr.length){
            return res;
        }
        if(sArr[index] == 'e' || sArr[index] == 'E'){
            index++;
            res = res && scanNum(sArr);
        }
        while(index < sArr.length && sArr[index] == ' ')
            index++;
        return res && index == sArr.length;
    }
    private boolean scanUnsignedNum(char[] s){
        if(index >= s.length){
            return false;
        }
        int before = index;
        while(index < s.length && s[index] >= '0' && s[index] <= '9'){
            index++;
        }
        return index > before;
    }
    private boolean scanNum(char[] s){
        if(index >= s.length){
            return false;
        }
        if(s[index] == '+' || s[index] == '-')
            index++;
        return scanUnsignedNum(s);
    }
}
```

# 22. 链表中倒数第k个节点
```
输入一个链表，输出该链表中倒数第k个节点。为了符合大多数人的习惯，本题从1开始计数，即链表的尾节点是倒数第1个节点。例如，一个链表有6个节点，从头节点开始，它们的值依次是1、2、3、4、5、6。这个链表的倒数第3个节点是值为4的节点。

 

示例：

给定一个链表: 1->2->3->4->5, 和 k = 2.

返回链表 4->5.

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/lian-biao-zhong-dao-shu-di-kge-jie-dian-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 注意
这道题的考点在于代码的鲁棒性，需要考虑到head为null以及链表长度小于k的情况（如果k是无符号整数还需要考虑k等于0的情况）

## 代码
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode getKthFromEnd(ListNode head, int k) {
        if (head == null) return null;
        ListNode node1 = head;
        ListNode node2 = node1;
        while (k-- > 1) {
            if (node2.next == null) return null;
            node2 = node2.next;
        }

        while (node2.next != null) {
            node2 = node2.next;
            node1 = node1.next;
        }

        return node1;
    }
}
```

# 24.反转链表
```
定义一个函数，输入一个链表的头节点，反转该链表并输出反转后链表的头节点。

 

示例:

输入: 1->2->3->4->5->NULL
输出: 5->4->3->2->1->NULL
 

限制：

0 <= 节点个数 <= 5000

 

注意：本题与主站 206 题相同：https://leetcode-cn.com/problems/reverse-linked-list/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fan-zhuan-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码（递归）
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode temp = reverseList(head.next);
        head.next.next = head;
        head.next = null;
        return temp;
    }
}
```

## 代码（迭代）
```java
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
class Solution {
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, pre = null, next = null;

        while (cur != null) {
            next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }

        return pre;
    }
}
```

# 26. 树的子结构
```
输入两棵二叉树A和B，判断B是不是A的子结构。(约定空树不是任意一个树的子结构)

B是A的子结构， 即 A中有出现和B相同的结构和节点值。

例如:
给定的树 A:

     3
    / \
   4   5
  / \
 1   2
给定的树 B：

   4 
  /
 1
返回 true，因为 B 与 A 的一个子树拥有相同的结构和节点值。

示例 1：

输入：A = [1,2,3], B = [3,1]
输出：false
示例 2：

输入：A = [3,4,5,1,2], B = [4,1]
输出：true
限制：

0 <= 节点个数 <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-de-zi-jie-gou-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSubStructure(TreeNode A, TreeNode B) {
        if (A == null || B == null) return false;
        return helper(A, B) || isSubStructure(A.left, B) || isSubStructure(A.right, B);
    }

    private boolean helper(TreeNode A, TreeNode B) {
        if (B == null) return true;
        if (A == null) return false;

        if (A.val != B.val) return false;

        return helper(A.left, B.left) && helper(A.right, B.right);
    }
}
```

# 28. 对称的二叉树
```
请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。

例如，二叉树 [1,2,2,3,4,4,3] 是对称的。

    1
   / \
  2   2
 / \ / \
3  4 4  3
但是下面这个 [1,2,2,null,3,null,3] 则不是镜像对称的:

    1
   / \
  2   2
   \   \
   3    3

 

示例 1：

输入：root = [1,2,2,3,4,4,3]
输出：true
示例 2：

输入：root = [1,2,2,null,3,null,3]
输出：false
 

限制：

0 <= 节点个数 <= 1000

注意：本题与主站 101 题相同：https://leetcode-cn.com/problems/symmetric-tree/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public boolean isSymmetric(TreeNode root) {
        return isSymmetric(root, root);
    }

    private boolean isSymmetric(TreeNode node1, TreeNode node2) {
        if (node1 == null && node2 == null) return true;
        if (node1 == null || node2 == null) return false;
        if (node1.val != node2.val) return false;

        return isSymmetric(node1.left, node2.right) && isSymmetric(node1.right, node2.left);
    }
}
```

# 29. 顺时针打印矩阵
```
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字。

 

示例 1：

输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
示例 2：

输入：matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
输出：[1,2,3,4,8,12,11,10,9,5,6,7]
 

限制：

0 <= matrix.length <= 100
0 <= matrix[i].length <= 100
注意：本题与主站 54 题相同：https://leetcode-cn.com/problems/spiral-matrix/

通过次数1,836提交次数4,266

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shun-shi-zhen-da-yin-ju-zhen-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class Solution {
    public int[] spiralOrder(int[][] matrix) {
         if(matrix==null||matrix.length==0||matrix[0].length==0){
             return new int[0];
         }
         int[] answer=new int[matrix.length * matrix[0].length];
         int rl=0,rh=matrix.length-1;
         int cl=0,ch=matrix[0].length-1;
         int cur=0;
         while(rl<=rh&&cl<=ch){
             for(int j=cl;j<=ch;j++){
                 answer[cur++]=matrix[rl][j];
             }
             for(int i=rl+1;i<=rh;i++){
                 answer[cur++]=matrix[i][ch];
             }
             if(rl<rh&&cl<ch){
                for(int j=ch-1;j>=cl;j--){
                     answer[cur++]=matrix[rh][j];
                 }
                 for(int i=rh-1;i>=rl+1;i--){
                     answer[cur++]=matrix[i][cl];
                 } 
             rl++;
             cl++;
             rh--;
             ch--;
         }
         return answer;
    }
}

```

# 30. 包含min函数的栈
```
定义栈的数据结构，请在该类型中实现一个能够得到栈的最小元素的 min 函数在该栈中，调用 min、push 及 pop 的时间复杂度都是 O(1)。

 

示例:

MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.min();   --> 返回 -3.
minStack.pop();
minStack.top();      --> 返回 0.
minStack.min();   --> 返回 -2.
 

提示：

各函数的调用总次数不超过 20000 次

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/bao-han-minhan-shu-de-zhan-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class MinStack {
    Stack<Integer> st1, st2;
    /** initialize your data structure here. */
    public MinStack() {
        st1 = new Stack<>();
        st2 = new Stack<>();
    }
    
    public void push(int x) {
        st1.push(x);
        if (st2.empty() || x < st2.peek()) st2.push(x);
        else st2.push(st2.peek());
    }
    
    public void pop() {
        st2.pop();
        st1.pop();
    }
    
    public int top() {
        return st1.peek();
    }
    
    public int min() {
        return st2.peek();
    }
}

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.min();
 */
```

 # 31. 栈的压入、弹出序列
 
 ```
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如，序列 {1,2,3,4,5} 是某栈的压栈序列，序列 {4,5,3,2,1} 是该压栈序列对应的一个弹出序列，但 {4,3,5,1,2} 就不可能是该压栈序列的弹出序列。

 

示例 1：

输入：pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
输出：true
解释：我们可以按以下顺序执行：
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
示例 2：

输入：pushed = [1,2,3,4,5], popped = [4,3,5,1,2]
输出：false
解释：1 不能在 2 之前弹出。
 

提示：

0 <= pushed.length == popped.length <= 1000
0 <= pushed[i], popped[i] < 1000
pushed 是 popped 的排列。
注意：本题与主站 946 题相同：https://leetcode-cn.com/problems/validate-stack-sequences/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zhan-de-ya-ru-dan-chu-xu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```
 

## 代码
```java
class Solution {
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        if (pushed.length != popped.length) return false;

        Stack<Integer> st = new Stack<>();
        int i = 0, j = 0;

        while (i < pushed.length && j < popped.length) {
            if (st.empty() || st.peek() != popped[j]) st.push(pushed[i++]);
            else {
                st.pop();
                j++;
            }
        }

        while (j < i) {
            if (st.peek() == popped[j]) {
                st.pop();
                j++;
            } else {
                break;
            }
        }

        return i == j;
    }
}
```

# 32 - II. 从上到下打印二叉树 II
```
从上到下按层打印二叉树，同一层的节点按从左到右的顺序打印，每一层打印到一行。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [9,20],
  [15,7]
]
 

提示：

节点总数 <= 1000
注意：本题与主站 102 题相同：https://leetcode-cn.com/problems/binary-tree-level-order-traversal/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-ii-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<List<Integer>> ret = new ArrayList<>();
        Queue<TreeNode> que = new LinkedList<>();

        que.offer(root);
        int nextLevel = 0, curLevel = 1;

        while (!que.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            while (curLevel-- > 0) {
                TreeNode node = que.poll();
                list.add(node.val);
                if (node.left != null) {
                    que.offer(node.left);
                    nextLevel++;
                }
                if (node.right != null) {
                    que.offer(node.right);
                    nextLevel++;
                }
            }
            ret.add(list);

            curLevel = nextLevel;
            nextLevel = 0;
        }

        return ret;
    }
}
```

# 32 - III. 从上到下打印二叉树 III
```
请实现一个函数按照之字形顺序打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右到左的顺序打印，第三行再按照从左到右的顺序打印，其他行以此类推。

 

例如:
给定二叉树: [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
返回其层次遍历结果：

[
  [3],
  [20,9],
  [15,7]
]
 

提示：

节点总数 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/cong-shang-dao-xia-da-yin-er-cha-shu-iii-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```


## 代码
```java
/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        if (root == null) return new ArrayList<>();
        List<List<Integer>> ret = new ArrayList<>();

        Stack<TreeNode> st1 = new Stack<>();
        Stack<TreeNode> st2 = new Stack<>();

        st1.push(root);

        while (!st1.empty() || !st2.empty()) {
            List<Integer> list = new ArrayList<>();
            if (!st1.empty()) {
                while (!st1.empty()) {
                    TreeNode node = st1.pop();
                    list.add(node.val);
                    if (node.left != null) st2.push(node.left);
                    if (node.right != null) st2.push(node.right);
                }
            } else {
                while (!st2.empty()) {
                    TreeNode node = st2.pop();
                    list.add(node.val);
                    if (node.right != null) st1.push(node.right);
                    if (node.left != null) st1.push(node.left);
                }
            }
            ret.add(list);
        }
        return ret;
    }
}
```

# 33. 二叉搜索树的后序遍历序列
```
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历结果。如果是则返回 true，否则返回 false。假设输入的数组的任意两个数字都互不相同。

 

参考以下这颗二叉搜索树：

     5
    / \
   2   6
  / \
 1   3
示例 1：

输入: [1,6,3,2,5]
输出: false
示例 2：

输入: [1,3,2,6,5]
输出: true
 

提示：

数组长度 <= 1000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-de-hou-xu-bian-li-xu-lie-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码

```java
class Solution {
    public boolean verifyPostorder(int[] postorder) {
        return helper(postorder, 0, postorder.length - 1);
    }

    public boolean helper(int[] postorder, int start, int end) {
        if (end <= start) return true;

        int pivotIndex = -1;
        for (int i = start; i < end; i++) {
            if (pivotIndex == -1 && postorder[i] > postorder[end]) pivotIndex = i;
            if ((pivotIndex != -1) && (postorder[i] < postorder[end])) return false;
        }

        if (pivotIndex == -1) return helper(postorder, start, end - 1);
        else return helper(postorder, start, pivotIndex - 1) && helper(postorder, pivotIndex, end - 1);
    }
}
```

# 35. 复杂链表的复制
```
请实现 copyRandomList 函数，复制一个复杂链表。在复杂链表中，每个节点除了有一个 next 指针指向下一个节点，还有一个 random 指针指向链表中的任意节点或者 null。

 

示例 1：



输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
示例 2：



输入：head = [[1,1],[2,1]]
输出：[[1,1],[2,1]]
示例 3：



输入：head = [[3,null],[3,0],[3,null]]
输出：[[3,null],[3,0],[3,null]]
示例 4：

输入：head = []
输出：[]
解释：给定的链表为空（空指针），因此返回 null。
 

提示：

-10000 <= Node.val <= 10000
Node.random 为空（null）或指向链表中的节点。
节点数目不超过 1000 。
 

注意：本题与主站 138 题相同：https://leetcode-cn.com/problems/copy-list-with-random-pointer/

 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/fu-za-lian-biao-de-fu-zhi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/
class Solution {
    public Node copyRandomList(Node head) {
        Node cur = head;
        while (cur != null) {
            Node temp = new Node(cur.val);
            temp.next = cur.next;
            cur.next = temp;
            cur = cur.next.next;
        }

        cur = head;
        while (cur != null) {
            if (cur.random != null) cur.next.random = cur.random.next;
            cur = cur.next.next;
        }

        cur = head;
        Node dummy = new Node(-1);
        Node dummyCur = dummy;
        while (cur != null) {
            dummyCur.next = cur.next;
            dummyCur = dummyCur.next;
            cur.next = cur.next.next;
            cur = cur.next;
        }

        return dummy.next;
    }
}
```

# 36. 二叉搜索树与双向链表

```
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的循环双向链表。要求不能创建任何新的节点，只能调整树中节点指针的指向。

 

为了让您更好地理解问题，以下面的二叉搜索树为例：

 



 

我们希望将这个二叉搜索树转化为双向循环链表。链表中的每个节点都有一个前驱和后继指针。对于双向循环链表，第一个节点的前驱是最后一个节点，最后一个节点的后继是第一个节点。

下图展示了上面的二叉搜索树转化成的链表。“head” 表示指向链表中有最小元素的节点。

 



 

特别地，我们希望可以就地完成转换操作。当转化完成以后，树中节点的左指针需要指向前驱，树中节点的右指针需要指向后继。还需要返回链表中的第一个节点的指针。

 

注意：本题与主站 426 题相同：https://leetcode-cn.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/

注意：此题对比原题有改动。

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/er-cha-sou-suo-shu-yu-shuang-xiang-lian-biao-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
/*
// Definition for a Node.
class Node {
    public int val;
    public Node left;
    public Node right;

    public Node() {}

    public Node(int _val) {
        val = _val;
    }

    public Node(int _val,Node _left,Node _right) {
        val = _val;
        left = _left;
        right = _right;
    }
};
*/
class Solution {
    public Node treeToDoublyList(Node root) {
        if (root == null) return null;

        Node pre, cur = root, curPre = null;;
        Node first = null, last = null;

        while (cur != null) {
            if (cur.left == null) {
                if (first == null) first = cur;
                cur.left = curPre;
                if (curPre != null) curPre.right = cur;
                curPre = cur;
                last = cur;
                cur = cur.right;
            } else {
                pre = cur.left;

                while (pre.right != null && pre.right != cur) pre = pre.right;

                if (pre.right == null) {
                    pre.right = cur;
                    cur = cur.left;
                } else {
                    if (curPre != null) curPre.right = cur;
                    cur.left = curPre;
                    curPre = cur;
                    last = cur;
                    cur = cur.right;
                }
            }
        }

        first.left = last;
        last.right = first;
        return first;
    }
}
```

# 39. 数组中出现次数超过一半的数字
```
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。

 

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

 

示例 1:

输入: [1, 2, 3, 2, 2, 2, 5, 4, 2]
输出: 2
 

限制：

1 <= 数组长度 <= 50000

 

注意：本题与主站 169 题相同：https://leetcode-cn.com/problems/majority-element/

 

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-zu-zhong-chu-xian-ci-shu-chao-guo-yi-ban-de-shu-zi-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码1
```java
class Solution {
    public int majorityElement(int[] nums) {
        return majorityElement(nums, 0, nums.length - 1);
    }

    public int majorityElement(int[] nums, int start, int end) {
        if (end == start) return nums[start];

        int pivotIndex = start + new Random().nextInt(end - start + 1);
        swap(nums, pivotIndex, end);

        int parIndex = start;
        for (int i = start; i < end; i++) {
            if (nums[i] < nums[end]) swap(nums, i, parIndex++);
        }

        swap(nums, parIndex, end);
        int mid = (nums.length - 1) >>> 1;
        if (parIndex == mid) return nums[parIndex];
        else if (parIndex < mid) return majorityElement(nums, parIndex + 1, end);
        else return majorityElement(nums, start, parIndex - 1);
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }
}
```

## 代码2
```java
class Solution {
    public int majorityElement(int[] nums) {
        int count = 0, lastNum = -1;

        for (int i = 0; i < nums.length; i++) {
            if (count == 0) {
                lastNum = nums[i];
                count++;
            } else if (nums[i] == lastNum) {
                count++;
            } else {
                count--;
            }
        }

        return lastNum;
    }
}
```

# 40. 最小的k个数
```
输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

 

示例 1：

输入：arr = [3,2,1], k = 2
输出：[1,2] 或者 [2,1]
示例 2：

输入：arr = [0,1,2,1], k = 1
输出：[0]
 

限制：

0 <= k <= arr.length <= 10000
0 <= arr[i] <= 10000

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

```

## 代码 
```java
class Solution {
    public int[] getLeastNumbers(int[] arr, int k) {
        if (k == 0) return new int[0];
        getLeastNumbers(arr, 0, arr.length - 1, k - 1);
        int[] ret = new int[k];
        for (int i = 0; i < k; i++) {
            ret[i] = arr[i];
        }

        return ret;
    }

    public void getLeastNumbers(int[] arr, int start, int end, int k) {

        if (start == end) return;

        swap(arr, start + new Random().nextInt(end - start + 1), end);
        int parIndex = start;
        for (int i = start; i < end; i++) {
            if (arr[i] < arr[end]) swap(arr, i, parIndex++);
        }

        swap(arr, parIndex, end);

        if (parIndex == k) return;
        if (parIndex < k) getLeastNumbers(arr, parIndex + 1, end, k);
        else getLeastNumbers(arr, start, parIndex - 1, k);
    }

    public void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
```

# 43. 1～n整数中1出现的次数
```
输入一个整数 n ，求1～n这n个整数的十进制表示中1出现的次数。

例如，输入12，1～12这些整数中包含1 的数字有1、10、11和12，1一共出现了5次。

 

示例 1：

输入：n = 12
输出：5
示例 2：

输入：n = 13
输出：6
 

限制：

1 <= n < 2^31
注意：本题与主站 233 题相同：https://leetcode-cn.com/problems/number-of-digit-one/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/1nzheng-shu-zhong-1chu-xian-de-ci-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```


## 代码
```java
class Solution {
    // 以百位上1出现的次数为例，分三种情况讨论（高位为百位左边的数字， 低位为百位右边的数字）
    // 1. 若百位上为1，百位上出现1的数字共有 高位 * 100 + 低位 + 1 个
    // 2。若百位上为0，百位上出现1的数字共有 高位 * 100 个
    // 3. 若百位上大于1, 百位上出现1的数字共有 (高位 + 1) * 100 个

    public int countDigitOne(int n) {
        long i = 1;
        int sum = 0;

        while (n / i != 0) {
            long high = n / i / 10;
            long low = n - (n / i) * i;
            long cur = (n / i) % 10;

            if (cur == 0) sum += high * i;
            else if (cur == 1) sum += high * i + low + 1;
            else sum += (high + 1) * i;

            i *= 10;
        }

        return sum;
    }
}
```

# 41. 数据流中的中位数
```
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

例如，

[2,3,4] 的中位数是 3

[2,3] 的中位数是 (2 + 3) / 2 = 2.5

设计一个支持以下两种操作的数据结构：

void addNum(int num) - 从数据流中添加一个整数到数据结构中。
double findMedian() - 返回目前所有元素的中位数。
示例 1：

输入：
["MedianFinder","addNum","addNum","findMedian","addNum","findMedian"]
[[],[1],[2],[],[3],[]]
输出：[null,null,null,1.50000,null,2.00000]
示例 2：

输入：
["MedianFinder","addNum","findMedian","addNum","findMedian"]
[[],[2],[],[3],[]]
输出：[null,null,2.00000,null,2.50000]
 

限制：

最多会对 addNum、findMedia进行 50000 次调用。
注意：本题与主站 295 题相同：https://leetcode-cn.com/problems/find-median-from-data-stream/

来源：力扣（LeetCode）
链接：https://leetcode-cn.com/problems/shu-ju-liu-zhong-de-zhong-wei-shu-lcof
著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
```

## 代码
```java
class MedianFinder {
    PriorityQueue<Integer> maxHeap, minHeap;
    int curIndex;

    /** initialize your data structure here. */
    public MedianFinder() {
        minHeap = new PriorityQueue<>();
        maxHeap = new PriorityQueue<>((i1, i2) -> i2 - i1);
        curIndex = 0;
    }
    
    public void addNum(int num) {
        if ((curIndex & 1) == 0) {
            if (maxHeap.isEmpty() || num >= maxHeap.peek()) {
                minHeap.offer(num);
            } else {
                maxHeap.offer(num);
                minHeap.offer(maxHeap.poll());
            }
        } else {
            if (minHeap.isEmpty() || num <= minHeap.peek()) {
                maxHeap.offer(num);
            } else {
                minHeap.offer(num);
                maxHeap.offer(minHeap.poll());
            }
        }

        curIndex++;
    }
    
    public double findMedian() {
        if ((curIndex & 1) == 0) {
            return (minHeap.peek() + maxHeap.peek()) / 2.0;
        } else {
            return (double)minHeap.peek();
        }
    }
}

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

