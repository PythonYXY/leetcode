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

