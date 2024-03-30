#include <iostream>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <climits>
#include <cstring>
#include <assert.h>

using namespace std;

int min_support;

struct FP_Node
{
    int item;
    int count;
    FP_Node *next_same;
    FP_Node *parent;
    map<int, FP_Node *> children;

    FP_Node(int item) : item(item) {}
};

struct Header
{
    int item;
    int frequency;
    FP_Node *head;
    FP_Node *tail;

    Header(int item, int frequency) : item(item), frequency(frequency)
    {
        head = new FP_Node(-1);
        tail = head;
    }
};

struct FP_Tree
{
    FP_Node *root;
    vector<Header> header_table;
    vector<vector<int>> transactions;

    FP_Tree(vector<Header> header_table, vector<vector<int>> transactions) : header_table(header_table), transactions(transactions)
    {
        root = new FP_Node(-1);
        root->count = 0;
        root->next_same = NULL;
        root->parent = NULL;
    }
};

bool compare(pair<int, int> &a, pair<int, int> &b)
{
    if (a.second != b.second)
        return a.second > b.second;
    else
        return a.first < b.first; // if same frequency, sorted by item number
}

vector<Header> ConstructHeaderTable(vector<vector<int>> &transactions)
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
        {
            item_count.erase(it++);
            // delete from transactions
            for (auto &transaction : transactions)
            {
                auto item = find(transaction.begin(), transaction.end(), (*it).first);
                if (item != transaction.end())
                    transaction.erase(item);
            }
        }
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

FP_Tree ConstructFPTree(vector<Header> header_table, vector<vector<int>> transactions)
{
    FP_Tree fp_tree(header_table, transactions);
    // scan the transactions to construct FP tree
    for (auto transaction : fp_tree.transactions)
    {
        FP_Node *cur_node = fp_tree.root;
        // construct tree path by frequency order
        for (auto header : fp_tree.header_table)
        {
            int item = header.item;
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

    return fp_tree;
}

bool isSinglePath(FP_Tree *fp_tree)
{
    if (fp_tree->root->children.size() != 0)
        return isSinglePath(fp_tree->root); // recursive call
    else
        return true; // nothing left
}

bool isSinglePath(FP_Node *fp_node)
{
}

int main(int argc, char *argv[])
{
    assert(argc == 4);
    double min_sup_ratio = atof(argv[1]);
    string input_file = argv[2];
    string output_file = argv[3];

    ifstream in(input_file);
    ofstream out(output_file);
    vector<string> lines;
    string transaction;

    while (getline(in, transaction))
    {
        cout << transaction;
    }

    cout << "hello world" << endl;

    return 0;
}