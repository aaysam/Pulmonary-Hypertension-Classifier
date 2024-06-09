PATH = "" # Enter your path to the CT scans

obs_num = 1
for observation in observations:
    print(observation)
    print(f'{obs_num}/34')
    obs_num += 1
    os.mkdir(f'/Users/iliakozhevnikov/med1_in_png/{observation}')
    for root, dirs, files in os.walk(f"{PATH}/{observation}"):
        for file in files:
            if file[-3:] == 'dcm':
                data = pydicom.dcmread(f'{root}/{file}')
                arr = data.pixel_array
                
                if len(arr.shape) == 2:
                    new_filename = f'{PATH}/med1_in_png/{observation}/{file.replace("dcm", "png")}'
                    plt.imsave(new_filename, arr, cmap="gray")
                
                elif len(arr.shape) == 3:
                    for CT_num in range(arr.shape[0]):
                        arr2d = arr[CT_num]
                        new_filename = f'{PATH}/med1_in_png/{observation}/{file.replace("dcm", "")}_{CT_num}.png'
                        plt.imsave(new_filename, arr2d, cmap="gray")
