from ytmusicapi import YTMusic
import pprint
import json

yt = YTMusic('../oauth.json')
# playlistId = yt.create_playlist('test', 'test description')
# search_results = yt.search('Oasis Wonderwall')
search_results = yt.get_charts('US')
# Explore the search_results object to find info on all the fields available.

def recurseObject(obj, indent=0):
    for key, value in obj.items():
        print(' ' * indent + key)
        if isinstance(value, dict):
            recurseObject(value, indent + 4)
        elif isinstance(value, list) or isinstance(value, tuple):
            for item in value:
                if isinstance(item, dict) or isinstance(item, list) or isinstance(item, tuple):
                    recurseObject(item, indent + 4)
                else:
                    print(' ' * (indent + 4) + str(item))
        else:
            print(' ' * (indent + 4) + str(value))

selection = input('Enter pprint, json, or recurse: ')
if selection == 'pprint':
    for i in range(0, 1):
        pprint.pprint(search_results)
        print('\n')
        print('\n')
        print('recursing search_results[' + str(i) + ']:')
        print('\n')
        recurseObject(search_results)
elif selection == 'json':
    for i in range(0, 2):
        print(json.dumps(search_results[i], indent=4))
else:
    for i in range(0, 2):
        print('search_results[' + str(i) + ']:')
        recurseObject(search_results[i])

