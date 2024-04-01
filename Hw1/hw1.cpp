#include <algorithm>
#include <assert.h>
#include <climits>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <string>
#include <typeinfo>
#include <vector>

/* Predefine */
using namespace std;
using Transaction = vector<int>;                // items
using FrequentPattern = pair<vector<int>, int>; // first: pattern, second: frequency

/* Global Parameters*/
int min_support;

/* Structs Define*/
struct FP_Node
{
    int item;
    int count;
    FP_Node *next_same;
    FP_Node *parent;
    map<int, FP_Node *> children;

    FP_Node(int item)
        : item(item)
    {
        parent = NULL;
        next_same = NULL;
    }
};

struct Header
{
    int item;
    int frequency;
    FP_Node *head;
    FP_Node *tail;

    Header(int item, int frequency)
        : item(item), frequency(frequency)
    {
        head = new FP_Node(-1);
        tail = head;
    }
};

struct FP_Tree
{
    FP_Node *root;
    vector<Header> header_table;
    vector<Transaction> transactions;

    FP_Tree(vector<Header> header_table, vector<Transaction> transactions)
        : header_table(header_table), transactions(transactions)
    {
        root = new FP_Node(-1);
        root->count = 0;
        root->next_same = NULL;
        root->parent = NULL;
    }
};

/* Functions Declare*/
vector<Header> ConstructHeaderTable(vector<Transaction> &transactions);
FP_Tree ConstructFPTree(vector<Header> &header_table, vector<Transaction> &transactions);
vector<FrequentPattern> FPGrowth(FP_Tree fp_tree);
vector<FrequentPattern> SinglePathFrequentPatterns(FP_Tree fp_tree);
vector<FrequentPattern> MultiPathFrequentPatterns(FP_Tree fp_tree);
bool isSinglePath(FP_Tree fp_tree);
bool isSinglePathRecur(FP_Node *fp_node);

/* Functions Impletment*/
bool compare(pair<int, int> &a, pair<int, int> &b)
{
    if (a.second != b.second)
        return a.second > b.second;
    else
        return a.first < b.first; // if same frequency, sorted by item number
}

vector<Header> ConstructHeaderTable(vector<Transaction> &transactions)
{
    map<int, int> item_count;
    vector<Header> header_table;

    // scan the transactions to count the frequency of each item
    for (auto transaction : transactions)
        for (auto item : transaction)
            item_count[item]++;

    // delete those frequency less than min support
    for (auto it = item_count.begin(); it != item_count.end();)
    {
        if ((*it).second < min_support)
            item_count.erase(it++);
        else
            it++;
    }

    // sort by their frequency in descending order
    vector<pair<int, int>> count_vector;
    for (auto item_freq_pair : item_count)
        count_vector.push_back(item_freq_pair);
    sort(count_vector.begin(), count_vector.end(), compare);

    // create header table according to sorted vector pair
    for (auto item_freq_pair : count_vector)
    {
        Header header(item_freq_pair.first, item_freq_pair.second);
        header_table.push_back(header);
    }

    return header_table;
}

FP_Tree ConstructFPTree(vector<Header> &header_table, vector<Transaction> &transactions)
{
    FP_Tree fp_tree(header_table, transactions);

    // scan the transactions to construct FP tree
    for (auto transaction : fp_tree.transactions)
    {
        FP_Node *cur_node = fp_tree.root;
        // construct tree path by frequency order
        for (auto &header : fp_tree.header_table)
        {
            int item = header.item;
            // find whether this item is in the current transaction
            if (find(transaction.begin(), transaction.end(), item) != transaction.end())
            {
                // find whether this item is in the current node's children
                auto it = cur_node->children.find(item);
                if (it == cur_node->children.end())
                {
                    // item doesn't exist -> create a new node
                    FP_Node *new_fp_node = new FP_Node(item);
                    new_fp_node->count = 1;
                    new_fp_node->parent = cur_node;
                    // update header link
                    header.tail->next_same = new_fp_node;
                    header.tail = new_fp_node;
                    // next node
                    cur_node->children[item] = new_fp_node;
                    cur_node = new_fp_node;
                }
                else
                {
                    // item already exist in chidren -> count frequency
                    (*it).second->count++;
                    cur_node = (*it).second;
                }
            }
        }
    }

    return fp_tree;
}

vector<FrequentPattern> FPGrowth(FP_Tree fp_tree)
{
    if (fp_tree.root->children.size() == 0)
        return {};
    if (isSinglePath(fp_tree))
        return SinglePathFrequentPatterns(fp_tree); // search frequent patterns on single path
    else
        return MultiPathFrequentPatterns(fp_tree); // search frequent patterns on multi path
}

vector<FrequentPattern> SinglePathFrequentPatterns(FP_Tree fp_tree)
{
    // there is only single path in the FP tree
    // so any frequent pattern will be generated from this path
    vector<FrequentPattern> single_path_patterns;
    FP_Node *cur_node = fp_tree.root->children.begin()->second;
    bool isFirstPattern = true;

    // search frequent patterns
    while (cur_node)
    {
        int item = cur_node->item;
        int count = cur_node->count;

        // check if this node is the first pattern to add
        if (isFirstPattern == false)
        {
            // if not, combine this node to the previous patterns
            int now_size = single_path_patterns.size(); // used for stop the under loop
            int now_count = 0;
            for (auto pattern : single_path_patterns)
            {
                FrequentPattern new_combine_pattern = pattern;
                new_combine_pattern.first.push_back(item);
                new_combine_pattern.second = count;
                single_path_patterns.push_back(new_combine_pattern);
                now_count++;
                if (now_count == now_size) // if not do this
                    break;                 // this loop will keep executing since we add a new pattern everytime
            }
        }

        // add only this node itself as a new pattern
        FrequentPattern new_pattern = {{item}, count};
        single_path_patterns.push_back(new_pattern);
        isFirstPattern = false;

        // keep seaching for new patterns
        if (cur_node->children.size() == 1) // double check single path and switch to next
            cur_node = cur_node->children.begin()->second;
        else // nothing left in this path
            cur_node = NULL;
    }

    return single_path_patterns;
}

vector<FrequentPattern> MultiPathFrequentPatterns(FP_Tree fp_tree)
{
    // there are multiple path, construct a conditional FP tree to search for frequent patterns
    vector<FrequentPattern> all_frequent_patterns;

    for (auto header = fp_tree.header_table.rbegin(); header != fp_tree.header_table.rend(); header++)
    {
        // traverse header table by inverse order, since we want to mining fp tree by less frequency first
        int item = (*header).item;
        int cur_frequency = 0;
        vector<Transaction> conditional_transactions;
        FP_Node *cur_node = (*header).head->next_same;

        // search all the path which the leave is this node
        while (cur_node)
        {
            // this path will based on current header node's frequency
            cur_frequency += cur_node->count;         // current node frequency will be used in frequent pattern
            for (int i = 0; i < cur_node->count; i++) // execute for frequency times to represent correct transactions frequency
            {
                FP_Node *path_node = cur_node->parent; // path will not contain current node, so begin from its parent
                Transaction now_path;
                if (path_node->parent)
                {
                    while (path_node->parent) // stop when we reach the root
                    {
                        now_path.push_back(path_node->item);
                        path_node = path_node->parent;
                    }
                    conditional_transactions.push_back(now_path);
                }
            }
            cur_node = cur_node->next_same; // search for another path
        }

        // construct a conditional FP tree based on conditional transactions
        vector<Header> conditional_header_table = ConstructHeaderTable(conditional_transactions);
        FP_Tree conditional_tree = ConstructFPTree(conditional_header_table, conditional_transactions);

        // get all frequent patterns based on conditional FP Tree
        vector<FrequentPattern> conditional_frequent_patterns = FPGrowth(conditional_tree); // recursive call

        // combine conditinoal frequent patterns with current node
        all_frequent_patterns.push_back({{item}, cur_frequency});
        for (auto pattern : conditional_frequent_patterns)
        {
            pattern.first.push_back(item);
            all_frequent_patterns.push_back(pattern);
        }
    }

    return all_frequent_patterns;
}

bool isSinglePathRecur(FP_Node *fp_node)
{
    int children = fp_node->children.size();
    if (children == 0) // leave node, no children
        return true;
    else if (children == 1) // one children, keep checking
        return isSinglePathRecur(fp_node->children.begin()->second);

    return false; // more than one children, this is not single path
}

bool isSinglePath(FP_Tree fp_tree)
{
    if (fp_tree.root->children.size() != 0)
        return isSinglePathRecur(fp_tree.root); // recursive check whether there is only one path
    else
        return true; // nothing left in the tree
}

/* Main Function */
int main(int argc, char *argv[])
{
    assert(argc == 4);
    double min_sup_ratio = atof(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    ifstream in(input_file);
    ofstream out(output_file);
    vector<Transaction> transactions;
    string transaction_str;
    int total_transactions = 0;

    // read transaction records from file and convert it to integer
    while (getline(in, transaction_str))
    {
        char *transaction_char = transaction_str.data();
        char *item = strtok(transaction_char, (char *)",");
        Transaction transaction;
        while (item != NULL)
        {
            transaction.push_back(atoi(item));
            item = strtok(NULL, (char *)",");
        }
        transactions.push_back(transaction);
        total_transactions++;
    }

    min_support = (int)ceil(transactions.size() * min_sup_ratio);

    // construct fp tree
    vector<Header> header_table = ConstructHeaderTable(transactions);
    FP_Tree fp_tree = ConstructFPTree(header_table, transactions);

    // search for frequent patterns
    vector<FrequentPattern> all_frequent_patterns = FPGrowth(fp_tree);

    // output frequent patterns
    for (auto pattern : all_frequent_patterns)
    {
        for (int i = 0; i < pattern.first.size(); i++)
        {
            out << pattern.first[i];
            if (i != pattern.first.size() - 1)
                out << ",";
        }
        float support = (float)pattern.second / (float)total_transactions;
        out << ":" << fixed << setprecision(4) << support << endl;
    }

    return 0;
}