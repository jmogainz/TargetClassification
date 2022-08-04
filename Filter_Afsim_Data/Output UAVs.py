# create a list of numbers 100 to 15000 with a step of 100
range_list = list(range(100, 15100, 100))
position_west = 79.46305555555556
position_north = 37.99675

with open('output.txt', 'w') as out:
    for i in range(len(range_list)):
        # out.write(f"platform MSHORAD-{i+1} MSHORAD\n")
        # out.write("    side blue\n")
        # out.write(f"    position {position_north}n {position_west}w altitude 2 m agl\n")
        # out.write("end_platform\n\n")
        out.write(f"platform UAV-SLOW-1-{i+1} UAV-SLOW\n")
        out.write("    side red\n")
        out.write("    route\n")
        out.write(f"        position 37.99675n {position_west}w altitude {range_list[i]+2} m agl\n")
        out.write(f"    end_route\n")
        out.write("end_platform\n\n")

        position_west += 0.0001
        position_north += 0.1