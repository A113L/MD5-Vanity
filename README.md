# MD5-Vanity
Python + OpenCL code for searching MD5 hashes with a specific substring

Usage:

`Usage: python3 md5_vanity_search.py <target_substring> <min_length> <max_length> [offset]`

Example:

`python3 md5_vanity_search.py cafe 7 7 4`

Will search for hashes with `cafe` starting at position 4 in the hash of static ?a?a?a?a?a?a?a hashcat mask.

`O%m$    : 9411cafec700f797bfbbd1bbdb1a93c1`
