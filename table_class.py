import cv2


class Plotter():
    
    def __init__(self, col_width=400, row_height=60, title="", font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.7, font_color=(0, 0, 0), border_color=(0, 0, 0), line_thickness=2, background_color=(200,200,200), opacity=0.5, start_x=1100, start_y=20):
        self.col_width = col_width                              
        self.row_height = row_height                           
        self.title = title                                     
        self.font = font                                       
        self.font_size = font_size                             
        self.font_color = font_color                           
        self.border_color = border_color                        
        self.line_thickness = line_thickness                   
        self.start_x = start_x                                  
        self.start_y = start_y                                 
        self.background_color = background_color               
        self.opacity = opacity                                  
        
    def plot_table(self, frame, cell_data, num_rows=2, num_columns=2): 

        if self.title == "" or self.title == None:
            end_x = self.start_x + num_columns * self.col_width
            end_y = self.start_y + num_rows * self.row_height

            cv2.rectangle(frame, (self.start_x, self.start_y), (end_x, end_y), self.border_color, self.line_thickness)
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.start_x, self.start_y), (end_x, end_y), self.background_color, -1)
            cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)

            for i in range(1, num_rows):
                y = self.start_y + i * self.row_height
                cv2.line(frame, (self.start_x, y), (end_x, y), self.border_color, self.line_thickness)

            
            for i in range(1, num_columns):
                x = self.start_x + i * self.col_width
                cv2.line(frame, (x, self.start_y), (x, end_y), self.border_color, self.line_thickness)

            for i in range(0, num_columns*num_rows):
                row_start_x = self.start_x + (i % num_columns) * self.col_width
                row_start_y = self.start_y + (i // num_columns) * self.row_height
                text_height = cv2.getTextSize(cell_data[i], self.font, self.font_size, self.line_thickness)[0][1]
                text_width = cv2.getTextSize(cell_data[i], self.font, self.font_size, self.line_thickness)[0][0]
                text_start_x = row_start_x + (self.col_width - text_width) // 2
                text_start_y = row_start_y + self.row_height - (self.row_height - text_height) // 2
                cv2.putText(frame, cell_data[i], (text_start_x, text_start_y), self.font, self.font_size, self.font_color, self.line_thickness)
        
        else:
            
            num_rows = num_rows + 1

            end_x = self.start_x + num_columns * self.col_width
            end_y = self.start_y + num_rows * self.row_height
            
            text_width = cv2.getTextSize(self.title, self.font, self.font_size, self.line_thickness)[0][0]
            text_height = cv2.getTextSize(self.title, self.font, self.font_size, self.line_thickness)[0][1]
            
            text_start_x = self.start_x + (end_x - self.start_x - text_width) // 2
            text_start_y = self.start_y + self.row_height - (self.row_height - text_height) // 2

            cv2.rectangle(frame, (self.start_x, self.start_y), (end_x, end_y), self.border_color, self.line_thickness)
            overlay = frame.copy()
            cv2.rectangle(overlay, (self.start_x, self.start_y), (end_x, end_y), self.background_color, -1)
            cv2.addWeighted(overlay, self.opacity, frame, 1 - self.opacity, 0, frame)

            
            cv2.putText(frame, self.title, (text_start_x, text_start_y), self.font, self.font_size, self.font_color, self.line_thickness)
            
            for i in range(1, num_rows):
                y = self.start_y + i * self.row_height
                cv2.line(frame, (self.start_x, y), (end_x, y), self.border_color, self.line_thickness)

            
            for i in range(1, num_columns):
                x = self.start_x + i * self.col_width
                cv2.line(frame, (x, self.start_y + self.row_height), (x, end_y), self.border_color, self.line_thickness)
            
            for i in range(0, num_columns*(num_rows-1)):
                row_start_x = self.start_x + (i % num_columns) * self.col_width
                row_start_y = self.start_y + self.row_height + (i // num_columns) * self.row_height
                text_height = cv2.getTextSize(cell_data[i], self.font, self.font_size, self.line_thickness)[0][1]
                text_width = cv2.getTextSize(cell_data[i], self.font, self.font_size, self.line_thickness)[0][0]
                text_start_x = row_start_x + (self.col_width - text_width) // 2
                text_start_y = row_start_y + self.row_height - (self.row_height - text_height) // 2
                cv2.putText(frame, cell_data[i], (text_start_x, text_start_y), self.font, self.font_size, self.font_color, self.line_thickness)