from data.graphs import Graphs
from data.chess import Chess


def get_dataset(args, tokenizer, device):
    if args.teacherless and tokenizer.name == 'numeral':
        teacherless_token = tokenizer.encode('$')[0]
    elif args.teacherless:
        teacherless_token = tokenizer.encode('$')[0]
    else:
        teacherless_token = None

    if args.dataset == 'chess':
        assert args.waypoint_len is None, "waypoint len is not useful for chess task!"
        data_path = './data/datasets/chess/mate_in_' + str(args.mate_in)
        train_path, test_path = data_path + '_train.json', data_path + '_test.json'
        train_data = Chess(tokenizer=tokenizer, n_samples=args.n_train, data_path=train_path, device=device,
                           teacherless_token=teacherless_token, unrolled=args.unrolled)
        test_data = Chess(tokenizer=tokenizer, n_samples=args.n_test, data_path=test_path, device=device,
                          teacherless_token=teacherless_token, unrolled=args.unrolled)

    elif args.dataset == 'graph':
        data_path = './data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + \
                    str(args.num_nodes)
        train_path, test_path = data_path + '_train_200000.txt', data_path + '_test_20000.txt'
        train_data = Graphs(tokenizer=tokenizer, n_samples=args.n_train, data_path=train_path, device=device,
                            teacherless_token=teacherless_token, reverse=args.reverse, waypoint_len=args.waypoint_len)
        if args.waypoint_len is not None and isinstance(args.waypoint_len, str):
            assert args.waypoint_len == "all", args.waypoint_len
            test_data = {}
            for waypoint_len in range(1, args.path_len):
                test_data_wp = Graphs(tokenizer=tokenizer, n_samples=args.n_test, data_path=test_path, device=device,
                                      teacherless_token=teacherless_token, reverse=args.reverse, waypoint_len=waypoint_len)
                test_data[f"waypoint_len_{waypoint_len}"] = test_data_wp
        else:
            test_data = Graphs(tokenizer=tokenizer, n_samples=args.n_test, data_path=test_path, device=device,
                               teacherless_token=teacherless_token, reverse=args.reverse, waypoint_len=args.waypoint_len)

    return train_data, test_data

