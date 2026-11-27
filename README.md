# Disk I/O and Storage Workload Analysis

**Topic 3** | **Python** | **Week 2 - In Progress**

Operating Systems Final Project

## 1. Project Overview

本專題使用 MSR Cambridge Block I/O Traces，
分析真實儲存系統的磁碟 I/O 行為，
探討讀寫工作負載、回應時間與存取區域性，
並結合作業系統相關理論進行分析與討論。

## 2. Research Objectives

- 分析 Read/Write 操作比例及工作負載特性。
- 探討 Request Size 與 Response Time 的關係。
- 分析磁碟存取模式與 Locality。
- 探討 Caching 與 OS I/O Management。

## 3. Research Questions

- RQ1: How do read and write workloads differ in
  operation frequency, request size, and I/O intensity?

- RQ2: How are operation type, request size, and
  workload intensity associated with response time?

- RQ3: What access locality patterns exist in the
  selected storage workloads?

## 4. Dataset

- Dataset: MSR Cambridge Block I/O Traces
- Selected Trace: To be confirmed
- Data Source:
  https://github.com/cacheMon/cache_dataset

[Dataset Documentation](./data/README.md)

## 5. Team Responsibilities

| Team | Members | Responsibilities |
| --- | --- | --- |
| Group 1 | 廖昌壘、莊澤洋 | Dataset and Read/Write Analysis |
| Group 2 | 余發宇、詹富期 | Response Time Analysis |
| Group 3 | 柯予恩、胡宇銜 | Storage Access Patterns |
| Project Integration | 林鼎恩 | Documentation, Website and Integration |

## 6. Project Structure

- [data/](./data/) - Dataset documentation
- [group1/](./group1/) - Read/Write workload analysis
- [group2/](./group2/) - Response time analysis
- [group3/](./group3/) - Storage access patterns
- [docs/](./docs/) - Project website and report

## 7. Project Schedule

| Week | Tasks | Status |
| --- | --- | --- |
| Week 1 | Dataset selection and project planning | To be verified |
| Week 2 | Data preprocessing and preliminary analysis | In Progress |
| Week 3 | Complete analysis and OS interpretation | Pending |
| Week 4 | Website, reproducibility and presentation | Pending |

## 8. Tools

- Python
- Pandas
- NumPy
- Matplotlib
- GitHub
- GitHub Pages

## 9. Reproducibility

All analysis scripts, dependencies, dataset information,
and execution instructions will be documented.

## 10. Project Website

GitHub Pages website is under development.

## 11. References

- [MSR Cambridge Trace Catalog](https://github.com/cacheMon/cache_dataset)

## 12. AI Assistance

ChatGPT is used to assist with project planning,
documentation, and technical discussion.
